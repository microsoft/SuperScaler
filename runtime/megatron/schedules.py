# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/schedules.py
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

from contextlib import contextmanager
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch.autograd.variable import Variable

from megatron import get_args
from megatron import get_num_microbatches
from megatron import get_timers
from megatron import mpu
from megatron import p2p_communication
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

def get_forward_backward_func():
    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def forward_step(forward_step_func, data_iterator, model, input_tensor, losses_reduced, extra_tensors = None):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
 
    timers = get_timers()
    timers('forward-compute').start()
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)

    output_tensor, output_extra_tensors, loss_func = forward_step_func(data_iterator, model, extra_tensors)

    if mpu.is_pipeline_last_stage():
        output_tensor = loss_func(output_tensor)
        loss, loss_reduced = output_tensor
        output_tensor = loss / get_num_microbatches()
        losses_reduced.append(loss_reduced)
    timers('forward-compute').stop()

    return output_tensor, output_extra_tensors

def update_output_extra_tensors_grad(output_extra_tensors, output_extra_tensors_grad):

    if output_extra_tensors_grad is not None and output_extra_tensors is not None:
        for key in sorted(output_extra_tensors):
            output_extra_tensors[key].grad = output_extra_tensors_grad[key]

def retain_input_tensors_grad_and_check_output_grad(input_tensor, extra_tensors, output_tensor, output_tensor_grad):

    if input_tensor is not None:
        for key in sorted(input_tensor):
            input_tensor[key].retain_grad()

    if extra_tensors is not None:
        for key in sorted(extra_tensors):
            if extra_tensors[key].requires_grad:
                extra_tensors[key].retain_grad()

    output_tensor_list = None
    output_tensor_grad_list = None

    if output_tensor is not None:
        if isinstance(output_tensor, dict):
            output_tensor_list = []
            for key in sorted(output_tensor):
                output_tensor_list.append(output_tensor[key])
        else:
            output_tensor_list = output_tensor

    if output_tensor_grad is not None:
        if isinstance(output_tensor_grad, dict):
            output_tensor_grad_list = []
            for key in sorted(output_tensor_grad):
                output_tensor_grad_list.append(output_tensor_grad[key])
        else:
            output_tensor_grad_list = output_tensor_grad

    return output_tensor_list, output_tensor_grad_list

def collect_grad_of_input_and_extra_tensors(input_tensor, extra_tensors):
    input_tensor_grad = None
    extra_tensors_grad = None
    if input_tensor is not None:
        input_tensor_grad = {}
        for key in sorted(input_tensor):
            if input_tensor[key].grad is None:
                input_tensor_grad[key] = torch.zeros(list(input_tensor[key].size()), requires_grad=False, device=torch.cuda.current_device(), dtype=torch.float16)
            else:
                input_tensor_grad[key] = input_tensor[key].grad

    # When we want to send back the gradients of some extra tensors (encoder_output),
    # its gradients may not be calculated yet, current workaround: send back zero values.
    if extra_tensors is not None:
        extra_tensors_grad = {}
        for key in sorted(extra_tensors):
            if extra_tensors[key].grad is None:
                extra_tensors_grad[key] = torch.zeros(list(extra_tensors[key].size()), requires_grad=False, device=torch.cuda.current_device(), dtype=torch.float16)
            else:
                extra_tensors_grad[key] = extra_tensors[key].grad

    return input_tensor_grad, extra_tensors_grad

def deallocate_output_tensor(out_dict):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if out_dict is None or isinstance(out_dict, torch.Tensor):
        return
    assert isinstance(out_dict, dict), f"dict of tensors is required. instead of: {out_dict}"
    for name in out_dict:
        assert isinstance(out_dict[name], torch.Tensor), \
            "expected Tensor, found %s." % type(out_dict[name]).__name__
        assert out_dict[name]._base is None, \
            f"counter-productive to free a view of another tensor. rank {torch.distributed.get_rank()}, tensor: {name}"
        out_dict[name].data = torch.empty(
            (1,),
            device = out_dict[name].device,
            dtype = out_dict[name].dtype,
    )
        

def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad, extra_tensors=None, output_extra_tensors=None, output_extra_tensors_grad=None, model=None):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    args = get_args()
    timers = get_timers()

    timers('backward-compute').start()

    update_output_extra_tensors_grad(output_extra_tensors, output_extra_tensors_grad)
    output_tensor, output_tensor_grad = retain_input_tensors_grad_and_check_output_grad(input_tensor, extra_tensors, output_tensor, output_tensor_grad)

    # Backward pass.
    if output_tensor_grad is None:       
        output_tensor = optimizer.scale_loss(output_tensor)
    
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    extra_tensors_grad = None

    input_tensor_grad, extra_tensors_grad = collect_grad_of_input_and_extra_tensors(input_tensor, extra_tensors)

    timers('backward-compute').stop()

    return input_tensor_grad, extra_tensors_grad

@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass

def forward_backward_no_pipelining(forward_step_func, data_iterator, model,
                                   optimizer, timers, forward_only):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    losses_reduced = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor, _= forward_step(forward_step_func, data_iterator, model,
                                         input_tensor, losses_reduced)

            if not forward_only:
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad, model=model)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor, _ = forward_step(forward_step_func, data_iterator, model,
                                 input_tensor, losses_reduced)

    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad, model=model)

    return losses_reduced


def forward_backward_pipelining_with_interleaving(forward_step_func, data_iterator, model,
                                                  optimizer, timers, forward_only):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []

    input_extra_tensors_list = [[] for _ in range(len(model))]
    output_extra_tensors_list = [[] for _ in range(len(model))] 

    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
        output_extra_tensor_grads_list = [[] for _ in range(len(model))]

    pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = mpu.get_pipeline_model_parallel_rank()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if get_num_microbatches() == pipeline_parallel_size:
            num_warmup_microbatches = num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = \
                (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (
                num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches,
                                          num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    def list_append_helper(tensor_list, model_chunk_id, new_tensor, name=None):
        tensor_list[model_chunk_id].append(new_tensor)      
    
    def list_pop_helper(tensor_list, model_chunk_id, name=None):
        popped_tensor = tensor_list[model_chunk_id].pop(0)

        return popped_tensor

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if mpu.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                list_append_helper(input_tensors, model_chunk_id, None, "input_tensor")
                list_append_helper(input_extra_tensors_list, model_chunk_id, None, "input_extra_tensors")
        input_tensor = input_tensors[model_chunk_id][-1]
        extra_tensors = input_extra_tensors_list[model_chunk_id][-1]

        output_tensor, output_extra_tensors = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     input_tensor, losses_reduced, extra_tensors)

        list_append_helper(output_tensors, model_chunk_id, output_tensor, "output_tensor")
        list_append_helper(output_extra_tensors_list, model_chunk_id, output_extra_tensors, "output_extra_tensors")

        return output_tensor, output_extra_tensors

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        mpu.set_virtual_pipeline_backward_model_parallel_rank(model_chunk_id)

        if mpu.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                list_append_helper(output_tensor_grads, model_chunk_id, None, "output_tensor_grads")
                list_append_helper(output_extra_tensor_grads_list, model_chunk_id, None, "output_extra_tensor_grads")

        input_tensor = list_pop_helper(input_tensors, model_chunk_id, "input_tensor")
        output_tensor = list_pop_helper(output_tensors, model_chunk_id, "output_tensor")
        output_tensor_grad = list_pop_helper(output_tensor_grads, model_chunk_id, "output_tensor_grad")   

        extra_tensors = list_pop_helper(input_extra_tensors_list, model_chunk_id, "input_extra_tensors")
        output_extra_tensors = list_pop_helper(output_extra_tensors_list, model_chunk_id, "output_extra_tensors")
        output_extra_tensors_grad = list_pop_helper(output_extra_tensor_grads_list, model_chunk_id, "output_extra_tensors_grad")

        input_tensor_grad, extra_tensors_grad = \
            backward_step(optimizer,  
                          input_tensor,
                          output_tensor,
                          output_tensor_grad,
                          extra_tensors,
                          output_extra_tensors,
                          output_extra_tensors_grad)

        return input_tensor_grad, extra_tensors_grad

    # Run warmup forward passes.
    mpu.set_virtual_pipeline_model_parallel_rank(0)
    mpu.set_virtual_pipeline_backward_model_parallel_rank(0)
    mpu.set_virtual_pipeline_next_forward_model_rank(0)
    mpu.set_virtual_pipeline_next_backward_model_rank(0)
    input_tensor, extra_tensors = p2p_communication.recv_forward(timers)

    list_append_helper(input_tensors, 0, input_tensor, "input_tensor")
    list_append_helper(input_extra_tensors_list, 0, extra_tensors, "input_extra_tensors")

    for k in range(num_warmup_microbatches):
        output_tensor, output_extra_tensors = forward_step_helper(k)
        
        current_forward_model_chunk_id = get_model_chunk_id(k, forward=True)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)

        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if mpu.is_pipeline_last_stage():
            output_tensor = None
            output_extra_tensors = None # number of extra tensors

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if k == (num_warmup_microbatches - 1) and not forward_only and not all_warmup_microbatches:
            input_tensor_grad = None
            extra_tensors_grad = None
            output_tensor_grad = None
            output_extra_tensors_grad = None
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False

            mpu.set_virtual_pipeline_model_parallel_rank(current_forward_model_chunk_id)
            mpu.set_virtual_pipeline_next_forward_model_rank(next_forward_model_chunk_id)
            mpu.set_virtual_pipeline_next_backward_model_rank(num_model_chunks-1)

            input_tensor, extra_tensors, output_tensor_grad, output_extra_tensors_grad = \
                p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor, output_extra_tensors, input_tensor_grad, extra_tensors_grad,
                        recv_prev=recv_prev, recv_next=recv_next, timers=timers)

            list_append_helper(output_tensor_grads, num_model_chunks-1, output_tensor_grad, "output_tensor_grads")
            list_append_helper(output_extra_tensor_grads_list, num_model_chunks-1, output_extra_tensors_grad, "output_extra_tensor_grads")

        else:
            mpu.set_virtual_pipeline_model_parallel_rank(current_forward_model_chunk_id)
            mpu.set_virtual_pipeline_next_forward_model_rank(next_forward_model_chunk_id)

            input_tensor, extra_tensors = \
                p2p_communication.send_forward_recv_forward(
                    output_tensor, output_extra_tensors, recv_prev, timers)

        list_append_helper(input_tensors, next_forward_model_chunk_id, input_tensor, "input_tensor")
        list_append_helper(input_extra_tensors_list, next_forward_model_chunk_id, extra_tensors, "input_extra_tensors")

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        output_tensor, output_extra_tensors = forward_step_helper(forward_k)

        # Backward pass.
        backward_k = k
        input_tensor_grad, extra_tensors_grad = backward_step_helper(backward_k)

        # Send output_tensor and input_tensor_grad, receive input_tensor
        # and output_tensor_grad.

        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        if mpu.is_pipeline_last_stage():
            output_tensor = None
            output_extra_tensors = None

        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        if mpu.is_pipeline_first_stage():
            input_tensor_grad = None
            extra_tensors_grad = None

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            # First stage is ahead of last stage by (pipeline_parallel_size - 1).
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                recv_prev = False
            next_forward_model_chunk_id += 1
        else:
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                             forward=True)

        recv_next = True
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                recv_next = False
            next_backward_model_chunk_id -= 1
        else:
            next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                              forward=False)

        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        mpu.set_virtual_pipeline_backward_model_parallel_rank(backward_model_chunk_id)
        mpu.set_virtual_pipeline_next_forward_model_rank(next_forward_model_chunk_id)
        mpu.set_virtual_pipeline_next_backward_model_rank(next_backward_model_chunk_id)

        # Communicate tensors.
        input_tensor, extra_tensors, output_tensor_grad, output_extra_tensors_grad= \
            p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor, output_extra_tensors, input_tensor_grad, extra_tensors_grad,
                    recv_prev=recv_prev, recv_next=recv_next, timers=timers)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            list_append_helper(input_tensors, next_forward_model_chunk_id, input_tensor, "input_tensor")
            list_append_helper(input_extra_tensors_list, next_forward_model_chunk_id, extra_tensors, "input_extra_tensors")          
        if recv_next:
            list_append_helper(output_tensor_grads, next_backward_model_chunk_id, output_tensor_grad, "output_tensor_grads")
            list_append_helper(output_extra_tensor_grads_list, next_backward_model_chunk_id, output_extra_tensors_grad, "output_extra_tensor_grads")

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if all_warmup_microbatches:
            output_tensor_grad, output_extra_tensors_grad = p2p_communication.recv_backward(timers)
            list_append_helper(output_tensor_grads, num_model_chunks-1, output_tensor_grad, "output_tensor_grads")
            list_append_helper(output_extra_tensor_grads_list, num_model_chunks-1, output_extra_tensors_grad, "output_extra_tensor_grads")

        for k in range(num_microbatches_remaining, num_microbatches):
            input_tensor_grad, extra_tensors_grad = backward_step_helper(k)
            backward_model_chunk_id = get_model_chunk_id(k, forward=False)
            next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False

            mpu.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            mpu.set_virtual_pipeline_backward_model_parallel_rank(backward_model_chunk_id)
            mpu.set_virtual_pipeline_next_backward_model_rank(next_backward_model_chunk_id)        
            output_tensor_grad, output_extra_tensors_grad = \
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, extra_tensors_grad, recv_next, timers)

            if recv_next:
                list_append_helper(output_tensor_grads, next_backward_model_chunk_id, output_tensor_grad, "output_tensor_grads")
                list_append_helper(output_extra_tensor_grads_list, next_backward_model_chunk_id, output_extra_tensors_grad, "output_extra_tensor_grads")              

    return losses_reduced

def delete_tensors(tensor_dict):
    if tensor_dict is not None:
        saved_keys = []
        for key in tensor_dict:
            saved_keys.append(key)
        for key in saved_keys:
            del tensor_dict[key]
        del tensor_dict

def forward_backward_pipelining_without_interleaving(forward_step_func, data_iterator,
                                                     model, optimizer, timers,
                                                     forward_only):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    input_tensors = []
    output_tensors = []
    losses_reduced = []

    input_extra_tensors_list = []
    output_extra_tensors_list = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):  
        input_tensor, extra_tensors = p2p_communication.recv_forward(timers)

        output_tensor, output_extra_tensors = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced, extra_tensors)

        p2p_communication.send_forward(output_tensor, output_extra_tensors, timers)

        if not forward_only: 
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            input_extra_tensors_list.append(extra_tensors)
            output_extra_tensors_list.append(output_extra_tensors)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:       
        input_tensor, extra_tensors = p2p_communication.recv_forward(timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):      
        last_iteration = (i == (num_microbatches_remaining - 1))

        output_tensor, output_extra_tensors = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced, extra_tensors)

        if forward_only:
            p2p_communication.send_forward(output_tensor, output_extra_tensors, timers)
        else:        
            output_tensor_grad, output_extra_tensors_grad = \
                p2p_communication.send_forward_recv_backward(output_tensor, output_extra_tensors, timers)

        # Add input_tensor and output_tensor to end of list, then pop from the
        # start of the list for backward pass.
        if not forward_only: 
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            input_extra_tensors_list.append(extra_tensors)
            output_extra_tensors_list.append(output_extra_tensors)     

        if forward_only:
            if not last_iteration:
                input_tensor, extra_tensors = p2p_communication.recv_forward(timers)
        else:
            input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
            extra_tensors, output_extra_tensors = input_extra_tensors_list.pop(0), output_extra_tensors_list.pop(0)

            input_tensor_grad, extra_tensors_grad = \
                backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad, extra_tensors, output_extra_tensors, 
                              output_extra_tensors_grad, model)

            delete_tensors(output_extra_tensors)
            delete_tensors(output_extra_tensors_grad)

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad, extra_tensors_grad, timers)
            else:            
                input_tensor, extra_tensors = \
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad, extra_tensors_grad, timers)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):        
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            extra_tensors = input_extra_tensors_list.pop(0)
            output_extra_tensors = output_extra_tensors_list.pop(0)

            output_tensor_grad, output_extra_tensors_grad = p2p_communication.recv_backward(timers)

            input_tensor_grad, extra_tensors_grad = \
                backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad, extra_tensors, output_extra_tensors, 
                              output_extra_tensors_grad, model)

            delete_tensors(output_extra_tensors)
            delete_tensors(output_extra_tensors_grad)            

            p2p_communication.send_backward(input_tensor_grad, extra_tensors_grad, timers)

    return losses_reduced
