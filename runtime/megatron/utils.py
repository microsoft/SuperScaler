# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/utils.py
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

"""General utilities."""

import sys

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import get_args
from megatron import print_rank_0
from megatron import get_adlr_autoresume
from megatron import mpu
from megatron.model.module import param_is_not_shared
from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate
from dataclasses import dataclass, field
import gc
import os
DEBUG_MEM = os.environ.get("DEBUG_MEM", '0') == '1'

@dataclass
class OpConfig:
    name: str
    prev_name: str
    input_tensors_info: dict = field(default_factory=dict)
    output_tensors_info: dict = field(default_factory=dict)
    input_extra_tensors_info: dict = field(default_factory=dict)
    output_extra_tensors_info: dict = field(default_factory=dict)    
    shared_weights_info: dict = field(default_factory=dict)

snapmap = dict()

def debug_mem_report(log_name, path=None, return_string=False):
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''
    
    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        string = ""
        # print('Storage on %s' %(mem_type))
        string += 'Storage on %s\n' %(mem_type)
        # print('-'*LEN)
        string += '-'*LEN + "\n"
        total_numel = 0
        total_mem = 0
        visited_data = []
        string_large = ""
        string_small = ""
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            # print('%s\t\t%s\t\t%.2f' % (element_type, size, mem) )
            if mem > 1:
                string_large += '%s\t\t%s\t\t%.2f\t\t%d\n' % (element_type, size, mem, data_ptr)
            else:
                string_small += '%s\t\t%s\t\t%.2f\t\t%d\n' % (element_type, size, mem, data_ptr)
        # print('-'*LEN)
        string += string_large
        # string += "\n" + string_small
        string += '-'*LEN + "\n"
        # print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        string += 'Total Tensors: %d \tUsed Memory Space: %.2f MBytes\n' % (total_numel, total_mem)
        # print('-'*LEN)
        string += '-'*LEN + "\n"
        return string

    string = ""
    LEN = 65
    string += f"================================== rank {torch.distributed.get_rank()} {log_name} ==================================\n"
    objects = gc.get_objects()
    string += '%s\t%s\t\t\t%s\n' %('Element type', 'Size', 'Used MEM(MBytes)')
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    string += _mem_report(cuda_tensors, 'GPU')
    string += '='*LEN + "\n"
    if path:
        with open(path, "a+") as f:
            f.write(string+"\n")   
    else:
        if return_string:
            return string
        else:
            print(string + "\n")        

def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name, get_list=False):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    allocated = torch.cuda.memory_allocated() / mega_bytes
    max_allocated = torch.cuda.max_memory_allocated() / mega_bytes
    reserved = torch.cuda.memory_reserved() / mega_bytes
    max_reserved = torch.cuda.max_memory_reserved() / mega_bytes

    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(allocated)
    string += ' | max allocated: {}'.format(max_allocated)
    string += ' | reserved: {}'.format(reserved)
    string += ' | max reserved: {}'.format(max_reserved)

    if get_list:
        mem_to_csv = [["allocated", "max_allocated", "reserved", "max_reserved"], 
            [f"{allocated:.2f}", f"{max_allocated:.2f}", f"{reserved:.2f}", f"{max_reserved:.2f}"]]
        return string, mem_to_csv

    return string


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, lr_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


