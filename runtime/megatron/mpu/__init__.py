# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/mpu/__init__.py
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

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy

from .data import broadcast_data

from .initialize import is_unitialized
from .initialize import destroy_model_parallel
from .initialize import get_data_parallel_group
from .initialize import get_data_parallel_rank
from .initialize import get_data_parallel_world_size
from .initialize import get_model_parallel_group
from .initialize import get_tensor_model_parallel_group
from .initialize import get_pipeline_model_parallel_group
from .initialize import get_tensor_model_parallel_rank, set_tensor_model_parallel_rank
from .initialize import get_pipeline_model_parallel_rank, set_pipeline_model_parallel_rank
from .initialize import is_pipeline_first_stage, is_pipeline_last_stage
from .initialize import get_tensor_model_parallel_src_rank
from .initialize import get_tensor_model_parallel_world_size, set_tensor_model_parallel_world_size
from .initialize import get_pipeline_model_parallel_world_size, set_pipeline_model_parallel_world_size
from .initialize import get_virtual_pipeline_model_parallel_rank, set_virtual_pipeline_model_parallel_rank
from .initialize import model_parallel_is_initialized

from .layers import ColumnParallelLinear
from .layers import RowParallelLinear
from .layers import VocabParallelEmbedding
from .layers import (set_tensor_model_parallel_attributes,
                     set_defaults_if_not_set_tensor_model_parallel_attributes,
                     copy_tensor_model_parallel_attributes)

from .layers import NewColumnParallelLinear     
from .layers import NewRowParallelLinear    
from .layers import NewVocabParallelEmbedding                  
from .cross_entropy import new_vocab_parallel_cross_entropy
from .mappings import _PrimReplicate
from .mappings import _PrimAllGather


from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region

from .random import checkpoint
from .random import get_cuda_rng_tracker
from .random import init_checkpointed_activations_memory_buffer
from .random import model_parallel_cuda_manual_seed
from .random import reset_checkpointed_activations_memory_buffer
from .random import gather_split_1d_tensor
from .random import split_tensor_into_1d_equal_chunks

from .utils import divide
from .utils import split_tensor_along_last_dim

from .initialize import initialize_model_parallel_flexpipe
from .initialize import get_stage_comm_recv_ranks
from .initialize import get_stage_comm_send_ranks
from .initialize import get_op_start_index
from .initialize import get_op_end_index
from .initialize import get_num_ops_list

from .initialize import set_virtual_pipeline_next_backward_model_rank
from .initialize import set_virtual_pipeline_next_forward_model_rank
from .initialize import get_virtual_pipeline_next_backward_model_rank
from .initialize import get_virtual_pipeline_next_forward_model_rank
from .initialize import get_virtual_pipeline_model_parallel_world_size
from .initialize import set_virtual_pipeline_backward_model_parallel_rank
from .initialize import get_virtual_pipeline_backward_model_parallel_rank
from .initialize import get_pipeline_rank_via_op_index
from .initialize import get_ranks_via_pipeline_stage
from .initialize import get_next_pipeline_model_parallel_rank
from .initialize import get_prev_pipeline_model_parallel_rank
from .initialize import set_comm_info
from .initialize import get_recv_info
from .initialize import get_send_info
from .initialize import get_group
from .initialize import get_op_dp_size
from .initialize import get_op_tp_size
from .initialize import set_resharding_group
from .initialize import get_resharding_group
from .initialize import set_resharding_rank
from .initialize import get_resharding_rank
from .initialize import set_resharding_dim
from .initialize import get_resharding_dim
from .initialize import get_data_parallel_group_via_op_index
from .initialize import get_tensor_model_parallel_group_via_op_index
from .initialize import set_op_resharding_ranks
from .initialize import get_op_resharding_ranks
from .initialize import get_tensor_model_parallel_ranks_via_op_index
from .initialize import get_data_parallel_ranks_via_op_index