# coding=utf-8
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

from abc import ABC
from abc import abstractmethod

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.utils import unwrap_model
from .module import Float16Module

import os
LOG_NAME = os.environ.get("LOG_NAME", None)

class MemoryBuffer:

    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.zeros(self.numel,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)


    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor



class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module


    @abstractmethod
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)



class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():
                self._grad_buffers[dtype] = MemoryBuffer(num_elements, dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.     

            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)
        
        args = get_args()
        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        self.resharding = args.resharding_stages[rank_in_pipeline]

    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad.data is not None:
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()

    ## TODO: continious buffer with resharding.
    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        # args = get_args()
        if self._grad_buffers is not None:
            if self.resharding:
                raise RuntimeError("cross-op resharding with continues buffer is not supported yet.")
            for _, buffer_ in self._grad_buffers.items():
                buffer_.data /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    buffer_.data, group=mpu.get_data_parallel_group())
        else:
            if self.resharding:
                # Otherwise, bucketize and all-reduce
                buckets = {}
                dp_groups = {}
                dp_sizes = {}
                # Pack the buckets.
                model_ = unwrap_model(self.module, (Float16Module)) 
                for op in model_.language_model.ops:
                    tp_size = op.tp_size
                    dp_size = op.dp_size
                    for param in op.parameters():
                        if param.requires_grad and param.grad is not None:
                            data_type = param.data.type()
                            key_str = str(data_type)+str(tp_size)+str(dp_size)
                            if key_str not in buckets:
                                buckets[key_str] = []
                            buckets[key_str].append(param)
                            param.main_grad = param.grad

                            if key_str not in dp_groups:
                                dp_groups[key_str] = mpu.get_data_parallel_group_via_op_index(op.op_index)
                                dp_sizes[key_str] = dp_size

                # For each bucket, all-reduce and copy all-reduced grads.
                for key_str in buckets:
                    bucket = buckets[key_str]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= dp_sizes[key_str]
                    torch.distributed.all_reduce(
                        coalesced, group=dp_groups[key_str])
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)
            else:
                # Otherwise, bucketize and all-reduce
                buckets = {}
                # Pack the buckets.
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = param.data.type()
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                        param.main_grad = param.grad

                # For each bucket, all-reduce and copy all-reduced grads.
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        coalesced, group=mpu.get_data_parallel_group())
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)                


    # def allreduce_gradients(self):
    #     """Reduce gradients across data parallel ranks."""
    #     # If we have buffers, simply reduce the data in the buffer.
    #     if self._grad_buffers is not None:
    #         for _, buffer_ in self._grad_buffers.items():
    #             buffer_.data /= mpu.get_data_parallel_world_size()
    #             torch.distributed.all_reduce(
    #                 buffer_.data, group=mpu.get_data_parallel_group())
    #     else:
    #         # Otherwise, bucketize and all-reduce
    #         buckets = {}
    #         # Pack the buckets.
    #         for param in self.module.parameters():
    #             if param.requires_grad and param.grad is not None:
    #                 tp = param.data.type()
    #                 if tp not in buckets:
    #                     buckets[tp] = []
    #                 buckets[tp].append(param)
    #                 param.main_grad = param.grad

    #         # print(f"[DEBUG] ======> allreduce_gradients <=====")
    #         # for name, params in self.module.named_parameters():
    #         #     if params.requires_grad:
    #         #         if params.grad is not None:
    #         #             string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad},\n main_grad: {params.main_grad}"
    #         #         else:
    #         #             string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad},\n grad = None"
    #         #     else:
    #         #         string = f"[DEBUG] param name {name}, requires_grad: {params.requires_grad}"
    #         #     with open(f"{LOG_NAME}_debug_grad_rank_{torch.distributed.get_rank()}.log", "a+") as f:
    #         #         f.write(string+"\n")  


    #         # For each bucket, all-reduce and copy all-reduced grads.
    #         for tp in buckets:
    #             bucket = buckets[tp]
    #             grads = [param.grad.data for param in bucket]
    #             coalesced = _flatten_dense_tensors(grads)
    #             coalesced /= mpu.get_data_parallel_world_size()
    #             torch.distributed.all_reduce(
    #                 coalesced, group=mpu.get_data_parallel_group())
    #             for buf, synced in zip(grads, _unflatten_dense_tensors(
    #                     coalesced, grads)):
    #                 buf.copy_(synced)
