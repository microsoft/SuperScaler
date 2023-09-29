# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/mpu/initialize.py
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


"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility
from megatron import get_args

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# For FlexPipe
_NUM_OPS_IN_EACH_STAGE_LIST =None
_OPS_START_INDEX_LIST = None
_OPS_END_INDEX_LIST = None

_CHILD_RANKS = None
_PARENT_RANKS = None

_FLEXPIPE_PREV_RANKS = None
_FLEXPIPE_NEXT_RANKS = None

_VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK = None

_RANKS_IN_EACH_PIPELINE_STAGE = None

_BWD_SEND_INFO = None
_FWD_RECV_INFO = None
_FWD_SEND_INFO = None
_BWD_RECV_INFO = None

all_groups = {}
_TP_SIZE_PER_OP = None
_DP_SIZE_PER_OP = None
_RESHARDING_GROUP = None
_RESHARDING_RANK = None
_RESHARDING_DIM = None
_OP_RESHARDING_RANKS = []

_TENSOR_MODEL_PARALLEL_RANKS = None
_DATA_PARALLEL_RANKS = None

def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None

def initialize_model_parallel_flexpipe():
    """
    Initialize model data parallel groups for FlexPipe.
    Generate _DATA_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP, _TENSOR_MODEL_PARALLEL_GROUP, _PIPELINE_MODEL_PARALLEL_GROUP in this function.
    Because FlexPipe supports different tensor model parallelism size at each pipeline stage,
    this function is quite different from original Megatron.
    """
    args = get_args()
    num_ops_in_each_stage = args.num_ops_in_each_stage
    virtual_pipeline_model_parallel_size_ = args.virtual_pipeline_model_parallel_size

    global _TP_SIZE_PER_OP, _DP_SIZE_PER_OP
    _TP_SIZE_PER_OP = []
    for i in range(len(args.model_parallel_size_of_each_op)):
        _TP_SIZE_PER_OP += args.model_parallel_size_of_each_op[i]
    _DP_SIZE_PER_OP = [] 
    for i in range(len(args.data_parallel_size_of_each_op)):
        _DP_SIZE_PER_OP += args.data_parallel_size_of_each_op[i]

    input_mp_size_of_each_stage = []
    output_mp_size_of_each_stage = []
    input_dp_size_of_each_stage = []
    output_dp_size_of_each_stage = []    

    if torch.distributed.get_rank() == 0:
        print('> initializing FlexPipe...')

    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size_    

    global _NUM_OPS_IN_EACH_STAGE_LIST
    _NUM_OPS_IN_EACH_STAGE_LIST = list(map(int, num_ops_in_each_stage))

    global _OPS_START_INDEX_LIST
    global _OPS_END_INDEX_LIST
    start_index = 0
    start_index_list = []
    end_index_list = []
    for i in range(len(_NUM_OPS_IN_EACH_STAGE_LIST)):
        start_index_list.append(start_index)
        start_index += _NUM_OPS_IN_EACH_STAGE_LIST[i]
        end_index_list.append(start_index)
    _OPS_START_INDEX_LIST = start_index_list
    _OPS_END_INDEX_LIST = end_index_list

    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    pipeline_model_parallel_size = len(_NUM_OPS_IN_EACH_STAGE_LIST)
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pipeline_model_parallel_size

    global _DATA_PARALLEL_GROUP, _DATA_PARALLEL_RANKS
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'    

    _DATA_PARALLEL_GROUP = []
    _DATA_PARALLEL_RANKS = []
    for i in range(pipeline_model_parallel_size):
        start_rank = 0
        for ii in range(0, i):
            STAGE_TP_SIZE = _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[ii]]
            STAGE_DP_SIZE = _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[ii]]
            start_rank += STAGE_TP_SIZE * STAGE_DP_SIZE
        end_rank = start_rank + _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[i]] * _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[i]]
        for op_index in range(_OPS_START_INDEX_LIST[i], _OPS_END_INDEX_LIST[i]):
            OP_TP_SIZE = _TP_SIZE_PER_OP[op_index]
            OP_DP_SIZE = _DP_SIZE_PER_OP[op_index]
            for j in range(OP_TP_SIZE):
                ranks = range(start_rank + j, end_rank, OP_TP_SIZE)
                group = get_group(ranks)
                if rank in ranks:
                    _DATA_PARALLEL_GROUP.append(group)
                    _DATA_PARALLEL_RANKS.append(ranks)

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP, _TENSOR_MODEL_PARALLEL_RANKS
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    _TENSOR_MODEL_PARALLEL_GROUP = []
    _TENSOR_MODEL_PARALLEL_RANKS = []
    for i in range(pipeline_model_parallel_size):
        start_rank = 0
        for ii in range(i):
            STAGE_TP_SIZE = _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[ii]]
            STAGE_DP_SIZE = _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[ii]]
            start_rank += STAGE_TP_SIZE * STAGE_DP_SIZE
        for op_index in range(_OPS_START_INDEX_LIST[i], _OPS_END_INDEX_LIST[i]):
            OP_TP_SIZE = _TP_SIZE_PER_OP[op_index]
            OP_DP_SIZE = _DP_SIZE_PER_OP[op_index]
            for j in range(OP_DP_SIZE):
                ranks = range(start_rank + j * OP_TP_SIZE, start_rank + (j+1) * OP_TP_SIZE)
                group = get_group(ranks)
                if rank in ranks:
                    _TENSOR_MODEL_PARALLEL_GROUP.append(group)
                    _TENSOR_MODEL_PARALLEL_RANKS.append(ranks)

    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    ranks_in_each_pipe_stage = []
    start_rank = 0
    for i in range(pipeline_model_parallel_size):
        STAGE_TP_SIZE = _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[i]]
        STAGE_DP_SIZE = _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[i]]
        end_rank = start_rank + STAGE_TP_SIZE * STAGE_DP_SIZE  
        ranks = [j for j in range(start_rank, end_rank)]
        if rank in ranks:
            _MPU_PIPELINE_MODEL_PARALLEL_RANK = i
        ranks_in_each_pipe_stage.append(ranks)
        start_rank = end_rank

    # store child ranks and parent ranks for each rank
    child_ranks = [[] for _ in range(world_size)]
    parent_ranks = [[] for _ in range(world_size)]

    stage_start_rank = 0
    for i in range(pipeline_model_parallel_size):
        if i != (pipeline_model_parallel_size -1):
            next_i = i + 1
        else:
            next_i = 0    
        tp_size = _TP_SIZE_PER_OP[_OPS_END_INDEX_LIST[i]-1]
        dp_size = _DP_SIZE_PER_OP[_OPS_END_INDEX_LIST[i]-1]
        tp_size_next = _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[next_i]]
        dp_size_next = _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[next_i]]

        for j in range(len(ranks_in_each_pipe_stage[i])):
            current_rank = ranks_in_each_pipe_stage[i][j]
            dp_id = j // tp_size
            tp_id = j % tp_size

            next_dp_id = [dp_id]
            next_tp_id = [tp_id]

            if tp_size_next > tp_size:
                ensure_divisibility(tp_size_next, tp_size)
                ratio = tp_size_next // tp_size
                next_tp_id = range(tp_id * ratio, (tp_id + 1)*ratio)
            if tp_size_next < tp_size:
                ensure_divisibility(tp_size, tp_size_next)
                ratio = tp_size // tp_size_next
                next_tp_id = [tp_id // ratio]          
            if dp_size_next > dp_size:
                ensure_divisibility(dp_size_next, dp_size)
                ratio = dp_size_next // dp_size
                next_dp_id = range(dp_id * ratio, (dp_id + 1)*ratio)
            if dp_size_next < dp_size:
                ensure_divisibility(dp_size, dp_size_next)
                ratio = dp_size // dp_size_next
                next_dp_id = [dp_id // ratio]           

            child_rank_list = []
            if next_i != 0:
                next_stage_start_index = stage_start_rank + len(ranks_in_each_pipe_stage[i])
            else:
                next_stage_start_index = 0
            for _dp_id in next_dp_id:
                for _tp_id in next_tp_id:
                    child_rank_list.append(next_stage_start_index + _dp_id * tp_size_next + _tp_id)
            child_ranks[current_rank] = child_rank_list
        
        stage_start_rank += len(ranks_in_each_pipe_stage[i])

    for i in range(pipeline_model_parallel_size):
        for j in range(len(ranks_in_each_pipe_stage[i])):
            current_rank = ranks_in_each_pipe_stage[i][j]
            for child_rank in child_ranks[current_rank]:
                parent_ranks[child_rank].append(current_rank)

    global _CHILD_RANKS
    global _PARENT_RANKS

    _CHILD_RANKS = child_ranks
    _PARENT_RANKS = parent_ranks

    global _FLEXPIPE_PREV_RANKS
    global _FLEXPIPE_NEXT_RANKS

    _FLEXPIPE_PREV_RANKS = parent_ranks[rank]
    _FLEXPIPE_NEXT_RANKS = child_ranks[rank]

    global _RANKS_IN_EACH_PIPELINE_STAGE
    _RANKS_IN_EACH_PIPELINE_STAGE = ranks_in_each_pipe_stage

    global _OP_RESHARDING_RANKS
    _OP_RESHARDING_RANKS = [None for _ in range(sum(_NUM_OPS_IN_EACH_STAGE_LIST))]

    ## fix: workaround for the group issue:
    if world_size >= 2:
        for i in range(0, world_size, 2):
            ranks = range(i, i+2)
            get_group(ranks)

    if world_size >= 4:
        for i in range(0, world_size, 4):
            ranks = range(i, i+4)
            get_group(ranks)    

    print(f'[DEBUG]|rank {torch.distributed.get_rank()}| \
    pipeline_rank= {get_pipeline_model_parallel_rank()} | \
    tp_size= {get_tensor_model_parallel_world_size()} | \
    tp_rank={get_tensor_model_parallel_rank()} | \
    tp_src_rank={get_tensor_model_parallel_src_rank()} | \
    dp_size= {get_data_parallel_world_size()} | \
    parent ranks={get_stage_comm_recv_ranks()} | \
    child ranks = {get_stage_comm_send_ranks()} | \
    args.micro_batch_size = {args.micro_batch_size}\n')

def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True

def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    return None
    # assert _MODEL_PARALLEL_GROUP is not None, \
    #     'model parallel group is not initialized'
    # return _MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_group(op_index=None):
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    args = get_args()
    if op_index is None:
        op_index = _OPS_START_INDEX_LIST[get_pipeline_model_parallel_rank()]
    return get_tensor_model_parallel_group_via_op_index(op_index)

def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    # assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
    #     'pipeline_model parallel group is not initialized'
    # return _PIPELINE_MODEL_PARALLEL_GROUP
    return None

def get_data_parallel_group(op_index=None):
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    args = get_args()
    if op_index is None:
        op_index = _OPS_START_INDEX_LIST[get_pipeline_model_parallel_rank()]
    return get_data_parallel_group_via_op_index(op_index)    

def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size

def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    assert _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None
    return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE

def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank

def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank

def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())

def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    assert _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None
    return _MPU_PIPELINE_MODEL_PARALLEL_RANK

def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if get_virtual_pipeline_model_parallel_world_size() is not None and \
            get_virtual_pipeline_model_parallel_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0

def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = \
            get_virtual_pipeline_model_parallel_world_size()
        if virtual_pipeline_model_parallel_world_size is not None and \
            get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1)

def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK

def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank

def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE

def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    return global_rank - get_tensor_model_parallel_rank()

def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())

def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None

def get_stage_comm_recv_ranks():
    assert _FLEXPIPE_PREV_RANKS is not None, \
        "_FLEXPIPE_PREV_RANKS is not initialized"    
    return _FLEXPIPE_PREV_RANKS

def get_stage_comm_send_ranks():
    assert _FLEXPIPE_NEXT_RANKS is not None, \
        "_FLEXPIPE_NEXT_RANKS is not initialized"    
    return _FLEXPIPE_NEXT_RANKS

def get_op_start_index(rank_in_pipeline, model_chunk_id=0):
    assert _OPS_START_INDEX_LIST is not None, \
        "_OPS_START_INDEX_LIST is not initialized"    
    num_pipeline_stages = len(_NUM_OPS_IN_EACH_STAGE_LIST)
    return _OPS_START_INDEX_LIST[rank_in_pipeline + model_chunk_id * num_pipeline_stages]    

def get_op_end_index(rank_in_pipeline, model_chunk_id=0):
    assert _OPS_END_INDEX_LIST is not None, \
        "_OPS_END_INDEX_LIST is not initialized"    
    num_pipeline_stages = len(_NUM_OPS_IN_EACH_STAGE_LIST)     
    return _OPS_END_INDEX_LIST[rank_in_pipeline + model_chunk_id * num_pipeline_stages]    

def get_num_ops_list():
    assert _NUM_OPS_IN_EACH_STAGE_LIST is not None, \
        "_NUM_OPS_IN_EACH_STAGE_LIST is not initialized"
    return _NUM_OPS_IN_EACH_STAGE_LIST

def set_virtual_pipeline_next_forward_model_rank(model_chunk_id):
    global _VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK = model_chunk_id

def set_virtual_pipeline_next_backward_model_rank(model_chunk_id):
    global _VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK = model_chunk_id

def get_virtual_pipeline_next_forward_model_rank():
    if _VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK is not None:
        return _VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK
    else:
        return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK

def get_virtual_pipeline_next_backward_model_rank():
    if _VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK is not None:
        return _VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK
    else:
        return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK

def set_virtual_pipeline_backward_model_parallel_rank(model_chunk_id):
    global _VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK = model_chunk_id

def get_virtual_pipeline_backward_model_parallel_rank():
    if _VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK is not None:
        return _VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK
    else:
        return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK

def get_pipeline_rank_via_op_index(op_index):
    global _NUM_OPS_IN_EACH_STAGE_LIST
    sum = 0
    for i in range(len(_NUM_OPS_IN_EACH_STAGE_LIST)):
        sum += _NUM_OPS_IN_EACH_STAGE_LIST[i]
        if sum > op_index:
            return  i % len(_NUM_OPS_IN_EACH_STAGE_LIST)

def get_ranks_via_pipeline_stage(pipeline_stage):
    return _RANKS_IN_EACH_PIPELINE_STAGE[pipeline_stage]

def get_next_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    if is_pipeline_last_stage():
        return 0
    else:
        return get_pipeline_model_parallel_rank() + 1

def get_prev_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    if is_pipeline_first_stage():
        return get_pipeline_model_parallel_world_size() - 1
    else:
        return get_pipeline_model_parallel_rank() - 1

def set_comm_info(bwd_send_info, fwd_recv_info, fwd_send_info, bwd_recv_info):
    global _BWD_SEND_INFO, _FWD_RECV_INFO, _FWD_SEND_INFO, _BWD_RECV_INFO
    _BWD_SEND_INFO = bwd_send_info
    _FWD_RECV_INFO = fwd_recv_info
    _FWD_SEND_INFO = fwd_send_info
    _BWD_RECV_INFO = bwd_recv_info

def get_recv_info(forward):
    global _FWD_RECV_INFO, _BWD_RECV_INFO
    if forward:
        return _FWD_RECV_INFO
    else:
        return _BWD_RECV_INFO

def get_send_info(forward):
    global _FWD_SEND_INFO, _BWD_SEND_INFO
    if forward:
        return _FWD_SEND_INFO
    else:
        return _BWD_SEND_INFO

def bitmap(ranks):
    """
    (from Zhiqi's codebase)
    map the rank list to the bit map string
    """
    bits = '0' * torch.distributed.get_world_size()
    for rank in ranks:
        if rank >= len(bits):
            raise ValueError("rank {} out of range ({})".format(rank, len(bits)))
        bits = bits[0:rank] + '1' + bits[rank+1:]
    return bits

def get_group(ranks):
    group_bits = bitmap(ranks)
    if group_bits not in all_groups: 
        all_groups[group_bits] = torch.distributed.new_group(list(ranks))       

    return all_groups[group_bits] 

def get_op_tp_size(op_index):
    return _TP_SIZE_PER_OP[op_index]

def get_op_dp_size(op_index):
    assert op_index < len(_DP_SIZE_PER_OP), f"op index {op_index} out of range({len(_DP_SIZE_PER_OP)})."
    return _DP_SIZE_PER_OP[op_index]

def set_resharding_group(devices):
    global _RESHARDING_GROUP
    _RESHARDING_GROUP = devices 

def get_resharding_group():
    global _RESHARDING_GROUP
    assert _RESHARDING_GROUP is not None
    return _RESHARDING_GROUP

def set_resharding_rank(rank):
    global _RESHARDING_RANK
    _RESHARDING_RANK = rank 

def get_resharding_rank():
    global _RESHARDING_RANK
    assert _RESHARDING_RANK is not None
    return _RESHARDING_RANK

def set_resharding_dim(dim):
    global _RESHARDING_DIM
    _RESHARDING_DIM = dim 

def get_resharding_dim():
    global _RESHARDING_DIM
    assert _RESHARDING_DIM is not None
    return _RESHARDING_DIM

def get_data_parallel_group_via_op_index(op_index):
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    pp_stage = get_pipeline_model_parallel_rank()
    start_op_index = _OPS_START_INDEX_LIST[pp_stage]
    return _DATA_PARALLEL_GROUP[op_index - start_op_index]

def get_tensor_model_parallel_group_via_op_index(op_index):
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'tensor model parallel group is not initialized'
    pp_stage = get_pipeline_model_parallel_rank()
    start_op_index = _OPS_START_INDEX_LIST[pp_stage]    
    return _TENSOR_MODEL_PARALLEL_GROUP[op_index - start_op_index]

def set_op_resharding_ranks(op_index, ranks):
    _OP_RESHARDING_RANKS[op_index] = ranks 

def get_op_resharding_ranks(op_index):
    return _OP_RESHARDING_RANKS[op_index]

def get_tensor_model_parallel_ranks_via_op_index(op_index):
    assert _TENSOR_MODEL_PARALLEL_RANKS is not None, \
        'tensor model parallel group is not initialized'
    pp_stage = get_pipeline_model_parallel_rank()
    start_op_index = _OPS_START_INDEX_LIST[pp_stage]    
    return _TENSOR_MODEL_PARALLEL_RANKS[op_index - start_op_index]    

def get_data_parallel_ranks_via_op_index(op_index):
    assert _DATA_PARALLEL_RANKS is not None, \
        'tensor model parallel group is not initialized'
    pp_stage = get_pipeline_model_parallel_rank()
    start_op_index = _OPS_START_INDEX_LIST[pp_stage]    
    return _DATA_PARALLEL_RANKS[op_index - start_op_index]        
