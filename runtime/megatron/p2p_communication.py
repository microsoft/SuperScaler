# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/p2p_communication.py
# Git commit hash: 42c1cf4279acea5a554500dcb552211f44cbec45
# We retain the following copyright from the original files:

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce
import operator
import torch

from megatron import get_args, get_timers
from megatron import mpu
from megatron.utils import debug_mem_report
from megatron.utils import report_memory

import os
import time

DEBUG_COMMUNICATE = os.environ.get("DEBUG_COMMUNICATE", '0') == '1'
EXTRA_TENSOR_TRANSFER = os.environ.get("EXTRA_TENSOR_TRANSFER", '1') == '1'

def print_tensor_dict_info(name, tensor_dict):
    args = get_args()
    string = f"rank {torch.distributed.get_rank()} {name} dict: \n"
    for key in sorted(tensor_dict):
        if tensor_dict[key] is not None:
            string += f"{key}: {list(tensor_dict[key].size())} size = {reduce(operator.mul, list(tensor_dict[key].size()), 1)}\n"
        else:
            string += f"{key}: {None}\n"

    with open(f"{args.log_path}{args.log_name}_debug_communicate_rank{torch.distributed.get_rank()}.log", "a+") as f:
        f.write(string+"\n") 

def print_communication_info(current_rank, op, other_rank, tensor_size):
    args = get_args()
    string = f"rank {current_rank} | {op} {other_rank}. size = {tensor_size}."
    with open(f"{args.log_path}{args.log_name}_debug_communicate_rank{current_rank}.log", "a+") as f:
        f.write(string+"\n")    

def _create_recv_placeholder(forward=True):
    args = get_args()
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float   

    recv_info = mpu.get_recv_info(forward)
    flatten_tensor_recv_prev = {}
    for key in sorted(recv_info["tensors"]):
        flatten_tensor_recv_prev[key] = []
        num_chunks = recv_info["tensors"][key]["num_tp_chunks"] * recv_info["tensors"][key]["num_dp_chunks"]
        recv_shape = list(recv_info["tensors"][key]["shape"])
        if recv_info["tensors"][key]["tp_split_dim"] == -1 and args.scatter_gather_tensors_in_pipeline:
            rank = mpu.get_pipeline_model_parallel_rank()
            if forward:
                op_index = mpu.get_op_start_index(rank)
            else:
                op_index = mpu.get_op_end_index(rank) - 1

            assert recv_shape[0] % mpu.get_op_tp_size(op_index) == 0
            recv_shape[0] //= mpu.get_op_tp_size(op_index)
            recv_shape[0] //= recv_info["tensors"][key]["num_tp_chunks"]
        for _ in range(num_chunks):
            flatten_tensor_recv_prev[key].append(torch.empty(recv_shape, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype))

    return flatten_tensor_recv_prev

def _partition(tensor, info, forward):
    """
    This function first partition each tensor and extra tensor according to number of receivers.
    Then flatten all the tensors and concat them into one large tensor.
    """
    
    tp_split_dim = info["tp_split_dim"]
    dp_split_dim = info["dp_split_dim"]
    num_tp_chunks = info["num_tp_chunks"]
    num_dp_chunks = info["num_dp_chunks"]
    tp_chunks_index = info["tp_chunks_index"]
    dp_chunks_index = info["dp_chunks_index"]
    args = get_args()

    if dp_split_dim != -1:
        _tmp_list = list(torch.chunk(tensor, chunks=num_dp_chunks, dim=dp_split_dim)) 
        tensor_split = []
        for i in range(len(dp_chunks_index)):
            tensor_split.append(_tmp_list[dp_chunks_index[i]].contiguous())
    else:
        tensor_split = [tensor for _ in range(num_dp_chunks)]
    
    if tp_split_dim != -1:
        for i in range(len(tensor_split)):
            _tmp_list = list(torch.chunk(tensor_split[i], chunks=num_tp_chunks, dim=tp_split_dim)) 
            tensor_split[i] = []
            for j in range(len(tp_chunks_index)):
                tensor_split[i].append(_tmp_list[tp_chunks_index[j]].contiguous())                
    else:
        for i in range(len(tensor_split)):
            if args.scatter_gather_tensors_in_pipeline:
                rank = mpu.get_pipeline_model_parallel_rank()
                if forward:
                    op_index = mpu.get_op_end_index(rank) - 1
                else:
                    op_index = mpu.get_op_start_index(rank)

                assert tensor_split[i].size()[0] >= num_tp_chunks * mpu.get_op_tp_size(op_index), "scatter_gather_tensors_in_pipeline is only available when mciro batch size >= num_splits"
                _tmp_list = list(torch.chunk(tensor_split[i], chunks=num_tp_chunks * mpu.get_op_tp_size(op_index), dim=0)) 
                tp_rank = torch.distributed.get_rank(group=mpu.get_tensor_model_parallel_group(op_index))
                new_tensor_split = [_tmp_list[num_tp_chunks * tp_rank + j].contiguous() for j in range(num_tp_chunks)]
            else:
                new_tensor_split = [tensor_split[i] for _ in range(num_tp_chunks)]
            tensor_split[i] = new_tensor_split

    _tensor_split = [n for a in tensor_split for n in a]

    return _tensor_split

def _reshape(recv_tensor, recv_info, forward):
    args = get_args()
    tensor_dict = {}
    extra_tensor_dict = {}

    for key in sorted(recv_info["tensors"]):
        num_tp_chunks = recv_info["tensors"][key]["num_tp_chunks"]
        num_dp_chunks = recv_info["tensors"][key]["num_dp_chunks"]
        tp_split_dim = recv_info["tensors"][key]["tp_split_dim"]
        dp_split_dim = recv_info["tensors"][key]["dp_split_dim"]
        tensor_list = recv_tensor[key]       

        if not EXTRA_TENSOR_TRANSFER and recv_info["tensors"][key]["extra_tensor"]:
            data_size = tensor_list[0].size()
            if args.model_name == "resnet":
                data_type = torch.float32
            else:
                data_type = torch.float16
            for i in range(len(tensor_list)):
                tensor_list[i] = torch.ones(data_size, requires_grad=True, device=torch.cuda.current_device(), dtype=data_type) 

        if num_tp_chunks > 1:
            if tp_split_dim == -1 and args.scatter_gather_tensors_in_pipeline:
                _tensor_list = []
                for i in range(len(tensor_list)):
                    _tensor_list.append(torch.cat(tensor_list[i: i+num_tp_chunks], dim=0))
                    i += num_tp_chunks
                tensor_list = _tensor_list  
            else:
                _tensor_list = []
                for i in range(len(tensor_list)):
                    _tensor_list.append(torch.cat(tensor_list[i: i+num_tp_chunks], dim=tp_split_dim))
                    i += num_tp_chunks
                tensor_list = _tensor_list  

        if num_dp_chunks > 1:
            _tensor_list = []
            for i in range(len(tensor_list)):
                _tensor_list.append(torch.cat(tensor_list[i: i+num_dp_chunks], dim=dp_split_dim))
                i += num_dp_chunks
            tensor_list = _tensor_list  

        if tp_split_dim == -1 and args.scatter_gather_tensors_in_pipeline:
            rank = mpu.get_pipeline_model_parallel_rank()
            if forward:
                op_index = mpu.get_op_start_index(rank)  
            else:
                op_index = mpu.get_op_end_index(rank) - 1
            tp_size = mpu.get_op_tp_size(op_index)

            gather_list = [torch.empty_like(tensor_list[0]) for _ in range(tp_size)]
            torch.distributed.all_gather(gather_list, tensor_list[0], group=mpu.get_tensor_model_parallel_group(op_index))
            output = torch.cat(gather_list, dim=0).contiguous()

            if recv_info["tensors"][key]["extra_tensor"]:
                extra_tensor_dict[key] = output
            else:
                tensor_dict[key] = output
        else:
            if recv_info["tensors"][key]["extra_tensor"]:
                extra_tensor_dict[key] = tensor_list[0]  
            else:
                tensor_dict[key] = tensor_list[0]

    if DEBUG_COMMUNICATE:
        print_tensor_dict_info("recieved tensors", tensor_dict)
        print_tensor_dict_info("received extra tensors", extra_tensor_dict)

    return tensor_dict, extra_tensor_dict

def _communicate_flexpipe(tensor_send_next, tensor_send_prev, extra_tensor_send_next, extra_tensor_send_prev, recv_prev, recv_next):

    timers = get_timers()

    prev_ranks = mpu.get_stage_comm_recv_ranks()
    next_ranks = mpu.get_stage_comm_send_ranks()
    num_parents = len(prev_ranks)
    num_childs = len(next_ranks)      
    tensor_recv_prev, extra_tensor_recv_prev, tensor_recv_next, extra_tensor_recv_next = None, None, None, None 

    # Create placeholder tensors for receive in forward and backward directions if needed.
    with torch.no_grad():
        if recv_prev:
            flatten_tensor_recv_prev = _create_recv_placeholder(forward=True)
        if recv_next:
            flatten_tensor_recv_next = _create_recv_placeholder(forward=False)

    if tensor_send_prev is not None:
        send_info = mpu.get_send_info(forward=False)
        for key in sorted(send_info["tensors"]):
            ops = []
            with torch.no_grad():
                if key in tensor_send_prev:
                    tensor_partitioned = _partition(tensor_send_prev[key], send_info["tensors"][key], forward=False)
                elif key in extra_tensor_send_prev:
                    if EXTRA_TENSOR_TRANSFER:
                        tensor_partitioned = _partition(extra_tensor_send_prev[key], send_info["tensors"][key], forward= False)
                    else:
                        continue
                else:
                    print(f"[rank {torch.distributed.get_rank()}] trying to send to prev, tensor name = {key}. send_info = {send_info['tensors']}")
            for i in range(num_parents):
                send_prev_op = torch.distributed.P2POp(torch.distributed.isend, tensor_partitioned[i], prev_ranks[i])
                ops.append(send_prev_op)  
                if DEBUG_COMMUNICATE:
                    print_communication_info(torch.distributed.get_rank(), f"send [{key} ({tensor_partitioned[i].dtype})] to ", prev_ranks[i], list(tensor_partitioned[i].size()))
            if recv_prev:
                recv_info = mpu.get_recv_info(forward=True)
                for i in range(num_parents):
                    recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, flatten_tensor_recv_prev[key][i], prev_ranks[i])
                    ops.append(recv_prev_op)
                    if DEBUG_COMMUNICATE:
                        print_communication_info(torch.distributed.get_rank(), f"recv [{key}] from ", prev_ranks[i], list(flatten_tensor_recv_prev[key][i].size()))                

            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            # torch.cuda.synchronize()
    elif recv_prev:
        recv_info = mpu.get_recv_info(forward=True)
        for key in sorted(recv_info["tensors"]): 
            if recv_info["tensors"][key]["extra_tensor"] and not EXTRA_TENSOR_TRANSFER:
                continue
            ops = []    
            for i in range(num_parents):
                recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, flatten_tensor_recv_prev[key][i], prev_ranks[i])
                ops.append(recv_prev_op)
                if DEBUG_COMMUNICATE:
                    print_communication_info(torch.distributed.get_rank(), f"recv [{key}] from ", prev_ranks[i], list(flatten_tensor_recv_prev[key][i].size()))

            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()  
            # torch.cuda.synchronize()        

    if tensor_send_next is not None:
        send_info = mpu.get_send_info(forward=True)
        for key in sorted(send_info["tensors"]):
            ops = []
            with torch.no_grad():
                if key in tensor_send_next:
                    tensor_partitioned = _partition(tensor_send_next[key], send_info["tensors"][key], forward=True)
                elif key in extra_tensor_send_next:
                    if EXTRA_TENSOR_TRANSFER:
                        tensor_partitioned = _partition(extra_tensor_send_next[key], send_info["tensors"][key], forward=True) 
                    else:
                        continue
            for i in range(num_childs):
                send_next_op = torch.distributed.P2POp(torch.distributed.isend, tensor_partitioned[i], next_ranks[i])
                ops.append(send_next_op)  
                if DEBUG_COMMUNICATE:
                    print_communication_info(torch.distributed.get_rank(), f"send [{key}] to ", next_ranks[i], list(tensor_partitioned[i].size()))
            if recv_next:
                recv_info = mpu.get_recv_info(forward=False)
                for i in range(num_childs):
                    recv_next_op = torch.distributed.P2POp(torch.distributed.irecv, flatten_tensor_recv_next[key][i], next_ranks[i])
                    ops.append(recv_next_op)
                    if DEBUG_COMMUNICATE:
                        print_communication_info(torch.distributed.get_rank(), f"recv [{key}] from ", next_ranks[i], list(flatten_tensor_recv_next[key][i].size()))                

            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            # torch.cuda.synchronize()

    elif recv_next:
        recv_info = mpu.get_recv_info(forward=False)
        for key in sorted(recv_info["tensors"]): 
            if recv_info["tensors"][key]["extra_tensor"] and not EXTRA_TENSOR_TRANSFER:
                continue            
            ops = []          
            for i in range(num_childs):
                recv_next_op = torch.distributed.P2POp(torch.distributed.irecv, flatten_tensor_recv_next[key][i], next_ranks[i])
                ops.append(recv_next_op)
                if DEBUG_COMMUNICATE:
                    print_communication_info(torch.distributed.get_rank(), f"recv [{key}] from ", next_ranks[i], list(flatten_tensor_recv_next[key][i].size()))  

            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()  
    # if len(ops) > 0:
    #     reqs = torch.distributed.batch_isend_irecv(ops)
    #     for req in reqs:
    #         req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    with torch.no_grad():
        if recv_prev:
            tensor_recv_prev, extra_tensor_recv_prev = _reshape(flatten_tensor_recv_prev, recv_info, forward=True)
        if recv_next:
            tensor_recv_next, extra_tensor_recv_next = _reshape(flatten_tensor_recv_next, recv_info, forward=False)

    if recv_prev:
        for key in sorted(tensor_recv_prev):
            tensor_recv_prev[key].requires_grad = True
        for key in sorted(extra_tensor_recv_prev):
            extra_tensor_recv_prev[key].requires_grad = True    
    if recv_next:
        for key in sorted(tensor_recv_next):
            tensor_recv_next[key].requires_grad = True
        for key in sorted(extra_tensor_recv_next):
            extra_tensor_recv_next[key].requires_grad = True                    

    return tensor_recv_prev, extra_tensor_recv_prev, tensor_recv_next, extra_tensor_recv_next

def recv_forward(timers=None):
    """Receive tensor from previous rank in pipeline (forward receive)."""  

    if mpu.is_pipeline_first_stage():
        input_tensors = None
        input_extra_tensors = None
    else:
        if timers is not None:
            timers('forward-recv').start()

        input_tensors, input_extra_tensors, _, _  = _communicate_flexpipe(
            tensor_send_next=None, 
            tensor_send_prev=None, 
            extra_tensor_send_next=None, 
            extra_tensor_send_prev=None, 
            recv_prev=True, 
            recv_next=False)
            
        if timers is not None:
            timers('forward-recv').stop()

    return input_tensors, input_extra_tensors


def recv_backward(timers=None):
    """Receive tensor from next rank in pipeline (backward receive)."""

    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
        output_extra_tensors_grad = None
    else:
        if timers is not None:
            timers('backward-recv').start()

        _, _, output_tensor_grad, output_extra_tensors_grad  = _communicate_flexpipe(
            tensor_send_next=None, 
            tensor_send_prev=None, 
            extra_tensor_send_next=None, 
            extra_tensor_send_prev=None, 
            recv_prev=False, 
            recv_next=True)

        if timers is not None:
            timers('backward-recv').stop()

    return output_tensor_grad, output_extra_tensors_grad


def send_forward(output_tensor, output_extra_tensors, timers=None):
    """Send tensor to next rank in pipeline (forward send)."""

    if not mpu.is_pipeline_last_stage():
        if timers is not None:
            timers('forward-send').start()

        _communicate_flexpipe(
            tensor_send_next=output_tensor, 
            tensor_send_prev=None, 
            extra_tensor_send_next=output_extra_tensors, 
            extra_tensor_send_prev=None, 
            recv_prev=False, 
            recv_next=False)

        if timers is not None:
            timers('forward-send').stop()


def send_backward(input_tensor_grad, extra_tensors_grad, timers=None):
    """Send tensor to previous rank in pipeline (backward send)."""

    if not mpu.is_pipeline_first_stage():
        if timers is not None:
            timers('backward-send').start()

        _communicate_flexpipe(
            tensor_send_next=None, 
            tensor_send_prev=input_tensor_grad, 
            extra_tensor_send_next=None, 
            extra_tensor_send_prev=extra_tensors_grad, 
            recv_prev=False, 
            recv_next=False)

        if timers is not None:
            timers('backward-send').stop()


def send_forward_recv_backward(output_tensor, output_extra_tensors, timers=None):
    """Batched send and recv with next rank in pipeline."""

    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
        output_extra_tensors_grad = None
    else:
        if timers is not None:
            timers('forward-send-backward-recv').start()

        _, _, output_tensor_grad, output_extra_tensors_grad = _communicate_flexpipe(
            tensor_send_next=output_tensor, 
            tensor_send_prev=None, 
            extra_tensor_send_next=output_extra_tensors, 
            extra_tensor_send_prev=None, 
            recv_prev=False, 
            recv_next=True)

        if timers is not None:
            timers('forward-send-backward-recv').stop()

    return output_tensor_grad, output_extra_tensors_grad


def send_backward_recv_forward(input_tensor_grad, extra_tensors_grad, timers=None):
    """Batched send and recv with previous rank in pipeline."""

    if mpu.is_pipeline_first_stage():
        input_tensor = None
        extra_tensors = None
    else:
        if timers is not None:
            timers('backward-send-forward-recv').start()

        input_tensor, extra_tensors, _, _ = _communicate_flexpipe(
            tensor_send_next=None, 
            tensor_send_prev=input_tensor_grad, 
            extra_tensor_send_next=None, 
            extra_tensor_send_prev=extra_tensors_grad, 
            recv_prev=True, 
            recv_next=False)

        if timers is not None:
            timers('backward-send-forward-recv').stop()

    return input_tensor, extra_tensors

def send_forward_recv_forward(output_tensor, output_extra_tensors, recv_prev, timers=None):
    """Batched recv from previous rank and send to next rank in pipeline."""

    if timers is not None:
        timers('forward-send-forward-recv').start()

    input_tensor, extra_tensors, _, _ = _communicate_flexpipe(
        tensor_send_next=output_tensor, 
        tensor_send_prev=None, 
        extra_tensor_send_next=output_extra_tensors, 
        extra_tensor_send_prev=None, 
        recv_prev=recv_prev, 
        recv_next=False)

    if timers is not None:
        timers('forward-send-forward-recv').stop()

    return input_tensor, extra_tensors


def send_backward_recv_backward(input_tensor_grad, extra_tensors_grad, recv_next, timers=None):
    """Batched recv from next rank and send to previous rank in pipeline."""

    if timers is not None:
        timers('backward-send-backward-recv').start()

    _, _, output_tensor_grad, output_extra_tensors_grad = _communicate_flexpipe(
        tensor_send_next=None, 
        tensor_send_prev=input_tensor_grad, 
        extra_tensor_send_next=None, 
        extra_tensor_send_prev=extra_tensors_grad, 
        recv_prev=False, 
        recv_next=recv_next)

    if timers is not None:
        timers('backward-send-backward-recv').stop()

    return output_tensor_grad, output_extra_tensors_grad


def send_forward_backward_recv_forward_backward(
        output_tensor, output_extra_tensors, input_tensor_grad, extra_tensors_grad, 
        recv_prev, recv_next, timers=None):
    """Batched send and recv with previous and next ranks in pipeline."""  

    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').start()

    input_tensor, extra_tensors, output_tensor_grad, output_extra_tensors_grad = _communicate_flexpipe(
        tensor_send_next=output_tensor, 
        tensor_send_prev=input_tensor_grad, 
        extra_tensor_send_next=output_extra_tensors, 
        extra_tensor_send_prev=extra_tensors_grad, 
        recv_prev=recv_prev, 
        recv_next=recv_next)

    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').stop()

    return input_tensor, extra_tensors, output_tensor_grad, output_extra_tensors_grad
