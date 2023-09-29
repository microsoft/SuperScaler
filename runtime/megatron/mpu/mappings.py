# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/mpu/mappings.py
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

import torch

from .initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, get_group, set_resharding_group, get_resharding_group, set_resharding_dim, get_resharding_dim, set_resharding_rank, get_resharding_rank, get_op_resharding_ranks, set_op_resharding_ranks, get_ranks_via_pipeline_stage, get_pipeline_model_parallel_rank
from .utils import split_tensor_along_last_dim, divide
import numpy as np

def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)

#### Aceso:

def new_split(input_, ranks, dim):

    if dim == -1:
        dim = input_.dim() - 1

    dim_size = divide(input_.size()[dim], len(ranks))
    tensor_list = torch.split(input_, dim_size, dim=dim)
    tensor_list = tuple(chunk.contiguous() for chunk in tensor_list)   
    return tensor_list[torch.distributed.get_rank(get_group(ranks))].contiguous()

def new_all_gather(input_, ranks, dim):
    if dim == -1:
        dim = input_.dim() - 1
    
    if not input_.is_contiguous():
        input_ = input_.contiguous()
    tensor_list = [torch.empty_like(input_) for _ in ranks]

    torch.distributed.all_gather(tensor_list, input_, group=get_group(ranks))
    torch.cuda.synchronize()

    # concat
    new_input_ = torch.cat(tensor_list, dim=dim).contiguous().requires_grad_()

    return new_input_    

def new_reduce(input_, ranks):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if len(ranks)==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_group(ranks))
    torch.cuda.synchronize()

    return input_

def new_reduce_scatter(input_, ranks, dim):

    input_list = list(input_.chunk(len(ranks), dim))
    for idx, tensor in enumerate(input_list):
        if not tensor.is_contiguous():
            input_list[idx] = tensor.contiguous()
    new_input_ = torch.empty_like(input_list[0], requires_grad=True)
    torch.distributed.reduce_scatter(new_input_, input_list, group=get_group(ranks))
    torch.cuda.synchronize()
    return new_input_

def new_all_to_all(input_, ranks, src_dim, dst_dim):

    input_list = list(input_.chunk(len(ranks), dim=dst_dim))
    for idx, tensor in enumerate(input_list):
        if not tensor.is_contiguous():
            input_list[idx] = tensor.contiguous()
    new_input_list = [torch.empty_like(t) for t in input_list]
    torch.distributed.all_to_all(new_input_list, input_list, group=get_group(ranks))
    torch.cuda.synchronize()
    new_input_ = torch.concat(tuple(new_input_list), dim=src_dim).requires_grad_()

    return new_input_    

class _PrimSplit(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_):
        # print(f"[DEBUG] fwd: split")
        ctx.ranks = get_resharding_group() 
        ctx.dim = get_resharding_dim()
        return new_split(input_, ctx.ranks, ctx.dim)

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"[DEBUG] bwd: all-gather")
        ranks = ctx.ranks
        dim = ctx.dim
        return new_all_gather(grad_output, ranks, dim)

class _PrimAllGather(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_):
        # print(f"[DEBUG] fwd: all-gather")
        ctx.ranks = get_resharding_group() 
        ctx.dim = get_resharding_dim()
        return new_all_gather(input_, ctx.ranks, ctx.dim)

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"[DEBUG] bwd: split")
        ranks = ctx.ranks
        dim = ctx.dim
        return new_split(grad_output, ranks, dim)

class _PrimAllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        # print(f"[DEBUG] fwd: all-reduce")
        ranks = get_resharding_group()
        return new_reduce(input_, ranks)

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"[DEBUG] bwd: None")
        return grad_output

class _PrimReduceScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        # print(f"[DEBUG] fwd: reduce-scatter")
        ctx.ranks = get_resharding_group()
        ctx.dim = get_resharding_dim()
        return new_reduce_scatter(input_, ctx.ranks, ctx.dim)

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"[DEBUG] bwd: all-gather")
        ranks = ctx.ranks
        dim = ctx.dim        
        return new_all_gather(grad_output, ctx.ranks, ctx.dim)

class _PrimReplicate(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        # print(f"[DEBUG] fwd: replicate")
        ctx.ranks = get_resharding_group()
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"[DEBUG] bwd: all-reduce")
        ranks = ctx.ranks
        return new_reduce(grad_output, ranks)

class _PrimAlltoAll(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        ctx.ranks = get_resharding_group()
        ctx.src_dim = get_resharding_dim()[0]
        ctx.dst_dim = get_resharding_dim()[1]
        return new_all_to_all(input_, ctx.ranks, ctx.src_dim, ctx.dst_dim)

    @staticmethod
    def backward(ctx, grad_output):
        ranks = ctx.ranks
        src_dim = ctx.src_dim
        dst_dim = ctx.dst_dim
        return new_all_to_all(grad_output, ranks, dst_dim, src_dim)

def prim_split(input_, ranks, dim):
    set_resharding_group(ranks)
    set_resharding_dim(dim)
    return _PrimSplit.apply(input_)

def prim_all_reduce(input_, ranks):
    set_resharding_group(ranks)
    return _PrimAllReduce.apply(input_)

def prim_reduce_scatter(input_, ranks, dim):
    set_resharding_group(ranks)
    set_resharding_dim(dim)
    return _PrimReduceScatter.apply(input_)

def prim_all_to_all(input_, ranks, src_dim, dst_dim):
    set_resharding_group(ranks)
    set_resharding_dim([src_dim, dst_dim])
    return _PrimAlltoAll.apply(input_)

def prim_all_gather(input_, ranks, dim):
    set_resharding_group(ranks)
    set_resharding_dim(dim)
    return _PrimAllGather.apply(input_)

def transpose(mat: np.ndarray, dim0: int, dim1: int, get_reverse=False):
    """
    (from Zhiqi's codebase)
    put the dim0 and dim1 of the mat to the last two dims
    """
    ndims = len(mat.shape)
    axes = list(range(ndims))
    assert dim0 < ndims and dim1 < ndims, "dim0 or dim1 out of index"
    axes.pop(max(dim0, dim1))
    axes.pop(min(dim0, dim1))
    axes += [dim0, dim1]

    if get_reverse:
        reverse_axes = []
        for original_index in range(ndims):
            for new_index in axes:
                if axes[new_index] == original_index:
                    reverse_axes.append(new_index)
        return np.transpose(mat, axes), reverse_axes
    else:
        return np.transpose(mat, axes)

def identical_spec(input_spec, required_spec):
    identical = True 
    ## this is used in T5, to pass encoder_output.
    if len(input_spec) == 0 and len(required_spec) == 0:
        return identical

    if input_spec["R"] != required_spec["R"]:
        identical = False
    if input_spec["V"] != required_spec["V"]:
        identical = False    
    for dim_index in range(len(input_spec["dims"])):
        if input_spec["dims"][dim_index] != required_spec["dims"][dim_index]:
            identical = False
    
    return identical

def tensor_adapter_handler(input_dev_mat, init_output_dev_mat, inc_dim, dec_dim, inc_to_size, dec_to_size):
    trans_in_dev_mat = transpose(input_dev_mat, inc_dim, dec_dim)
    trans_out_dev_mat, reverse_axes = transpose(init_output_dev_mat, inc_dim, dec_dim, get_reverse=True)

    for index_r in range(len(trans_in_dev_mat)): 
        for index_v in range(len(trans_in_dev_mat[index_r])): 
            for index_d in range(len(trans_in_dev_mat[index_r][index_v])):
                tmp_arrays = np.hsplit(trans_in_dev_mat[index_r][index_v][index_d], dec_to_size)
                tmp_arrays = [tmp_arrays[i].reshape(inc_to_size, 1) for i in range(len(tmp_arrays))]
                new_mat = np.hstack(tmp_arrays)
                trans_out_dev_mat[index_r][index_v][index_d] = new_mat
    output_dev_mat = trans_out_dev_mat.transpose(reverse_axes)   

    return trans_in_dev_mat, output_dev_mat

def tensor_adapter(input_, input_spec, required_spec, input_dev_mat):
    if identical_spec(input_spec, required_spec) or len(required_spec) == 0:
        return input_, input_dev_mat

    rank = torch.distributed.get_rank()
    # init_output_dev_mat = np.array([0 for _ in range(torch.distributed.get_world_size())]).reshape([required_spec["R"], required_spec["V"]] + required_spec["dims"])
    all_ranks = get_ranks_via_pipeline_stage(get_pipeline_model_parallel_rank())
    init_output_dev_mat = np.array(all_ranks).reshape([required_spec["R"], required_spec["V"]] + required_spec["dims"])

    if input_spec["R"] > required_spec["R"]:
        ## R -> Dim, split
        for dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][dim_index] < required_spec["dims"][dim_index]:
                assert input_spec["R"] % required_spec["R"] == 0

                trans_in_dev_mat, output_dev_mat = tensor_adapter_handler(
                    input_dev_mat, init_output_dev_mat, inc_dim=2+dim_index, dec_dim=0, 
                    inc_to_size=required_spec["dims"][dim_index], dec_to_size=required_spec["R"]
                )
                num_chunks = input_spec["R"] // required_spec["R"]

                for devices in trans_in_dev_mat.reshape(-1, num_chunks):
                    if rank in devices:
                        return prim_split(input_, devices, dim_index), output_dev_mat
                
    elif input_spec["V"] > required_spec["V"]:
        ## V -> R, all-reduce
        if input_spec["R"] < required_spec["R"]:
            assert input_spec["V"] % required_spec["V"] == 0

            trans_in_dev_mat, output_dev_mat = tensor_adapter_handler(
                input_dev_mat, init_output_dev_mat, inc_dim=0, dec_dim=1, 
                inc_to_size=required_spec["R"], dec_to_size=required_spec["V"]
            )
            num_chunks = input_spec["V"] // required_spec["V"]

            for devices in trans_in_dev_mat.reshape(-1, num_chunks):
                if rank in devices:
                    return prim_all_reduce(input_, devices), output_dev_mat    
            
        ## V-> D, reduce-scatter
        for dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][dim_index] < required_spec["dims"][dim_index]:
                assert input_spec["V"] % required_spec["V"] == 0

                trans_in_dev_mat, output_dev_mat = tensor_adapter_handler(
                    input_dev_mat, init_output_dev_mat, inc_dim=2+dim_index, dec_dim=1, 
                    inc_to_size=required_spec["dims"][dim_index], dec_to_size=required_spec["V"]
                )
                num_chunks = input_spec["V"] // required_spec["V"]

                for devices in trans_in_dev_mat.reshape(-1, num_chunks):
                    if rank in devices:
                        return prim_reduce_scatter(input_, devices, dim_index), output_dev_mat 

    else:
        for src_dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][src_dim_index] > required_spec["dims"][src_dim_index]:
                ## D -> R, all-gather
                if input_spec["R"] < required_spec["R"]:
                    assert input_spec["dims"][src_dim_index] % required_spec["dims"][src_dim_index] == 0
                    trans_in_dev_mat, output_dev_mat = tensor_adapter_handler(
                        input_dev_mat, init_output_dev_mat, inc_dim=0, dec_dim=2+src_dim_index, 
                        inc_to_size=required_spec["R"], dec_to_size=required_spec["dims"][src_dim_index]
                    )
                    num_chunks = input_spec["dims"][src_dim_index] // required_spec["dims"][src_dim_index]

                    for devices in trans_in_dev_mat.reshape(-1, num_chunks):
                        if rank in devices:                            
                            return prim_all_gather(input_, devices, src_dim_index), output_dev_mat  

                for dst_dim_index in range(len(input_spec["dims"])):
                    ## D -> D, all-to-all
                    if dst_dim_index != src_dim_index and input_spec["dims"][dst_dim_index] < required_spec["dims"][dst_dim_index]:
                        assert input_spec["dims"][src_dim_index] % required_spec["dims"][src_dim_index] == 0

                        trans_in_dev_mat, output_dev_mat = tensor_adapter_handler(
                            input_dev_mat, init_output_dev_mat, inc_dim=2+dst_dim_index, dec_dim=2+src_dim_index, 
                            inc_to_size=required_spec["dims"][dst_dim_index], dec_to_size=required_spec["dims"][src_dim_index]
                        )
                        num_chunks = input_spec["dims"][src_dim_index] // required_spec["dims"][src_dim_index]

                        for devices in trans_in_dev_mat.reshape(-1, num_chunks):
                            if rank in devices:
                                return prim_all_to_all(input_, devices, src_dim_index, dst_dim_index), output_dev_mat    

        raise RuntimeError(f"No communication pattern found. input_spec: {input_spec}\nrequired_spec: {required_spec}")

# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)

def copy_to_tensor_model_parallel_region_test(input_):
    # return _CopyToModelParallelRegion.apply(input_)
    return input_


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)

def new_copy_to_tensor_model_parallel_region(op_index, input_, input_spec, input_dev_mat):
    num_replicates = input_spec["R"]
    if num_replicates == 1:
        return input_ 
    else:
        op_resharding_ranks = get_op_resharding_ranks(op_index)
        if op_resharding_ranks is None:
            rank = torch.distributed.get_rank()
            trans_in_dev_mat = transpose(input_dev_mat, 1, 0)
            for ranks in trans_in_dev_mat.reshape(-1, num_replicates):
                if rank in ranks:
                    op_resharding_ranks = ranks 
                    set_op_resharding_ranks(op_index, ranks)
        set_resharding_group(op_resharding_ranks)
        return _PrimReplicate.apply(input_)
    
    raise RuntimeError("failed in new_copy_to_tensor_model_parallel_region")

def new_reduce_from_tensor_model_parallel_region(op_index, input_, input_spec, input_dev_mat):
    num_replicates = input_spec["R"]
    if num_replicates == 1:
        return input_ 
    else:
        op_resharding_ranks = get_op_resharding_ranks(op_index)
        if op_resharding_ranks is None:
            rank = torch.distributed.get_rank()
            trans_in_dev_mat = transpose(input_dev_mat, 1, 0)
            for ranks in trans_in_dev_mat.reshape(-1, num_replicates):
                if rank in ranks:
                    op_resharding_ranks = ranks 
                    set_op_resharding_ranks(op_index, ranks)                    
        set_resharding_group(op_resharding_ranks)
        return _PrimAllReduce.apply(input_)
    
    raise RuntimeError("failed in new_reduce_from_tensor_model_parallel_region")    