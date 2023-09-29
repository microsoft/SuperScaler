# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.utils import OpConfig
from .enums import AttnMaskType
from .utils import init_method_normal
from .utils import scaled_init_method_normal
from megatron.utils import debug_mem_report
from .flex_ops import gen_op
from .flex_model import get_flex_model
from megatron.model.enums import AttnMaskType, LayerType, AttnType

def _gpt(num_layers, init_method, output_layer_init_method, num_tokentypes, parallel_output, hidden_dropout, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels):
    op_list = []
    op_list.append({"name": "encoder-embedding", "op_index": 0, "embedding_dropout_prob": hidden_dropout, "init_method": init_method, "num_tokentypes": num_tokentypes, "prev_name": None})
    prev_name = "encoder-embedding"
    for i in range(num_layers):
        op_list.append({"name": "enc-1st-layernorm", "op_index": 1 + i * 13 + 0, "prev_name": prev_name})
        op_list.append({"name": "enc-attention-qkv", "op_index": 1 + i * 13 + 1, "init_method": init_method, "attention_type": AttnType.self_attn, "prev_name": "enc-1st-layernorm"})
        op_list.append({"name": "enc-attention-score", "op_index": 1 + i * 13 + 2, "layer_number": i, "prev_name": "enc-attention-qkv"})
        op_list.append({"name": "enc-attention-softmax", "op_index": 1 + i * 13 + 3, "layer_number": i, "attn_mask_type": AttnMaskType.causal, "prev_name": "enc-attention-score"})
        op_list.append({"name": "enc-attention-dropout", "op_index": 1 + i * 13 + 4, "prev_name": "enc-attention-softmax"})
        op_list.append({"name": "enc-attention-context", "op_index": 1 + i * 13 + 5, "prev_name": "enc-attention-dropout"})
        op_list.append({"name": "enc-attention-dense", "op_index": 1 + i * 13 + 6, "init_method": output_layer_init_method, "dim1_size": kv_channels * num_attention_heads, "dim2_size": hidden_size, "prev_name": "enc-attention-context"})
        op_list.append({"name": "enc-post-attention-dropout", "op_index": 1 + i * 13 + 7, "prev_name": "enc-attention-dense"})
        op_list.append({"name": "enc-2nd-layernorm", "op_index": 1 + i * 13 + 8, "prev_name": "enc-post-attention-dropout"})
        op_list.append({"name": "enc-MLP-GEMM-1", "op_index": 1 + i * 13 + 9, "init_method": init_method, "dim1_size": hidden_size, "dim2_size": ffn_hidden_size, "prev_name": "enc-2nd-layernorm"})
        op_list.append({"name": "enc-MLP-gelu", "op_index": 1 + i * 13 + 10, "prev_name": "enc-MLP-GEMM-1"})
        op_list.append({"name": "enc-MLP-GEMM-2", "op_index": 1 + i * 13 + 11, "init_method": output_layer_init_method, "dim1_size": ffn_hidden_size, "dim2_size": hidden_size, "prev_name": "enc-MLP-gelu"})
        op_list.append({"name": "enc-post-MLP-dropout", "op_index": 1 + i * 13 + 12, "prev_name": "enc-MLP-GEMM-2"})
        prev_name = "enc-post-MLP-dropout"

    op_list.append({"name": "final-layernorm", "op_index": 1 + num_layers * 13 + 0, "prev_name": prev_name})
    op_list.append({"name": "gpt-post-process", "op_index": 1 + num_layers * 13 + 1, "parallel_output": parallel_output, "init_method": init_method, "prev_name": "final-layernorm"})

    return op_list

class FlexGPTModel(MegatronModule):
    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True, 
                 profiling=False, hidden_dropout=0.1, num_layers=0, hidden_size=0, ffn_hidden_size=0, num_attention_heads=0, kv_channels=0):
        super(FlexGPTModel, self).__init__()
        args = get_args()
        init_method = init_method_normal(args.init_method_std)
        
        if not profiling:
            args = get_args()
            num_layers = args.num_layers
            output_layer_init_method = scaled_init_method_normal(args.init_method_std, num_layers)

            global op_start_index, op_end_index
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
            virtual_pipeline_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            op_start_index = mpu.get_op_start_index(pipeline_rank, virtual_pipeline_rank)
            op_end_index = mpu.get_op_end_index(pipeline_rank, virtual_pipeline_rank)
            num_ops = op_end_index - op_start_index
            assert num_ops >= 0

            full_op_list = _gpt(num_layers, init_method, output_layer_init_method, num_tokentypes, parallel_output, \
                                args.hidden_dropout, args.hidden_size, args.ffn_hidden_size, args.num_attention_heads, args.kv_channels)
            current_op_list = []
            for i in range(op_start_index, op_end_index):
                algo = args.algo_of_each_op[pipeline_rank][i-op_start_index]
                current_op_list.append(gen_op(full_op_list[i], algo))

            self.language_model = get_flex_model(
                full_model_op_list=current_op_list,
                pre_process=pre_process,
                post_process=post_process
                )
        else: ## generate op list for profiler
            output_layer_init_method = scaled_init_method_normal(args.init_method_std, num_layers)
            self.full_op_list = _gpt(num_layers, init_method, output_layer_init_method, num_tokentypes, parallel_output, \
                                hidden_dropout, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def set_extra_tensors(self, extra_tensors):
        """See megatron.model.transformer.set_extra_tensors()"""
        self.language_model.set_extra_tensors(extra_tensors)       

    def forward(self, input_tensors, input_extra_tensors):  

        lm_output = self.language_model(input_tensors, input_extra_tensors)
        return lm_output       