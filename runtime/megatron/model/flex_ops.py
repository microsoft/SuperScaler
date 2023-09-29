# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/model/transformer.py
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

"""Flexible ops definition for FlexPipe."""
import math
import torch
import torch.nn.functional as F
import os
import time 

from megatron import get_args
from megatron import mpu
from .module import RopalaModule
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu
from megatron.mpu.utils import ensure_divisibility
from megatron import get_timers
from .language_model import parallel_lm_logits, new_parallel_lm_logits
from megatron.utils import debug_mem_report
from megatron.utils import report_memory
from megatron.model.utils import init_method_normal
from functools import reduce
import operator
import numpy as np 
from megatron.mpu.mappings import tensor_adapter, new_copy_to_tensor_model_parallel_region, new_reduce_from_tensor_model_parallel_region
from megatron.mpu.mappings import copy_to_tensor_model_parallel_region, prim_all_gather, prim_all_reduce, prim_split 
from megatron.mpu.layers import set_tensor_model_parallel_attributes, _initialize_affine_weight_gpu
from megatron.model.utils import init_method_normal

DEBUG_OUTPUT = os.environ.get("DEBUG_OUTPUT", '0') == '1'

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)

def print_rank0(info):
    if torch.distributed.get_rank() == 0:
        print(info)

def gen_op(op_info, algo):
    if "conv" in op_info["name"]:
        ## conv1 only has 3 in_channels, cannot be partitioned.
        if op_info["name"] == "conv1":
            algo = 0
        op = ParallelConvOp(name=op_info["name"],
                            op_index = op_info["op_index"], 
                            input_width = op_info["input_width"],
                            in_channels = op_info["in_channels"], 
                            out_channels = op_info["out_channels"], 
                            kernel_size = op_info["kernel_size"], 
                            stride = op_info["stride"], 
                            padding = op_info["padding"], 
                            algo = algo, 
                            send_residual = op_info["send_residual"],
                            prev_name=op_info["prev_name"])
    elif "downsample" in op_info["name"]:
        op = ParallelConvOp(name=op_info["name"],
                            op_index = op_info["op_index"], 
                            input_width = op_info["input_width"],
                            in_channels = op_info["in_channels"], 
                            out_channels = op_info["out_channels"], 
                            kernel_size = op_info["kernel_size"], 
                            stride = op_info["stride"], 
                            padding = op_info["padding"], 
                            algo = algo, 
                            downsample=op_info["downsample"],
                            residual_in_channels = op_info["residual_in_channels"],
                            residual_input_width = op_info["residual_input_width"],
                            recv_residual=True,
                            prev_name=op_info["prev_name"])              
    elif "bn" in op_info["name"]:
        op = ParallelBatchNormOp(name=op_info["name"],
                            op_index = op_info["op_index"], 
                            input_width = op_info["input_width"],
                            in_channels = op_info["in_channels"], 
                            prev_name=op_info["prev_name"])    
    elif "relu" in op_info["name"]:
        op = ParallelReLUOp(name=op_info["name"],
                            op_index = op_info["op_index"], 
                            input_width = op_info["input_width"],
                            in_channels = op_info["in_channels"], 
                            prev_name=op_info["prev_name"])    
    elif "maxpool" in op_info["name"]:
        op = ParallelMaxPoolOp(name=op_info["name"],
                            op_index = op_info["op_index"], 
                            input_width = op_info["input_width"],
                            in_channels = op_info["in_channels"], 
                            kernel_size = op_info["kernel_size"], 
                            stride = op_info["stride"], 
                            padding = op_info["padding"],
                            prev_name=op_info["prev_name"])      
    elif "avgpool" in op_info["name"]:
        op = ParallelAdapAvgPoolOp(name=op_info["name"],
                            op_index = op_info["op_index"], 
                            input_width = op_info["input_width"],
                            in_channels = op_info["in_channels"], 
                            output_width = op_info["output_width"],
                            prev_name=op_info["prev_name"])     
    elif "fc" in op_info["name"]:
        op = ParallelFCOp(name=op_info["name"],
                            op_index = op_info["op_index"], 
                            input_width = op_info["input_width"],
                            in_channels = op_info["in_channels"], 
                            num_classes = op_info["num_classes"],
                            prev_name=op_info["prev_name"])   

    elif "embedding" in op_info["name"]:
        op = Embedding(name=op_info["name"], 
                        op_index = op_info["op_index"], 
                        embedding_dropout_prob = op_info["embedding_dropout_prob"],
                        init_method = op_info["init_method"],
                        num_tokentypes = op_info["num_tokentypes"],
                        prev_name=op_info["prev_name"])
    elif "layernorm" in op_info["name"]:
        op = ParallelLayerNormOp(name=op_info["name"], 
                                 op_index = op_info["op_index"], 
                                 prev_name=op_info["prev_name"])
    elif "attention-qkv" in op_info["name"]:
        op = ParallelQKVOp(name=op_info["name"], 
                            op_index = op_info["op_index"], 
                            init_method = op_info["init_method"],
                            attention_type = op_info["attention_type"],
                            algo = algo,
                            prev_name=op_info["prev_name"])
    elif "attention-score" in op_info["name"]:
        op = ParallelAttentionScoreOp(name=op_info["name"], 
                                        op_index = op_info["op_index"],
                                        layer_number = op_info["layer_number"], 
                                        prev_name=op_info["prev_name"])
    elif "attention-softmax" in op_info["name"]:
        op = ParallelSoftmaxFusionOp(name=op_info["name"], 
                                        op_index = op_info["op_index"],
                                        layer_number = op_info["layer_number"], 
                                        attn_mask_type = op_info["attn_mask_type"],
                                        prev_name=op_info["prev_name"])
    elif "post-attention-dropout" in op_info["name"] or "post-MLP-dropout" in op_info["name"]:
        op = ParallelDropoutOp(name=op_info["name"], 
                                 op_index = op_info["op_index"], 
                                 prev_name=op_info["prev_name"])   
    elif "attention-dropout" in op_info["name"]:
        op = ParallelSoftmaxDropoutOp(name=op_info["name"], 
                                 op_index = op_info["op_index"], 
                                 prev_name=op_info["prev_name"])
    elif "attention-context" in op_info["name"]:
        op = ParallelContextOp(name=op_info["name"], 
                                 op_index = op_info["op_index"], 
                                 prev_name=op_info["prev_name"])
    elif "dense" in op_info["name"] or "GEMM" in op_info["name"]:
        op = ParallelGEMM(name=op_info["name"], 
                            op_index = op_info["op_index"], 
                            dim1_size = op_info["dim1_size"],
                            dim2_size = op_info["dim2_size"],
                            init_method = op_info["init_method"],
                            prev_name=op_info["prev_name"])
    elif "gelu" in op_info["name"]:
        op = ParallelMlpGeLUOp(name=op_info["name"], 
                                 op_index = op_info["op_index"], 
                                 prev_name=op_info["prev_name"])
    elif "gpt-post-process" in op_info["name"]:
        op = GPTPostProcess(name=op_info["name"], 
                                 op_index = op_info["op_index"], 
                                 parallel_output = op_info["parallel_output"],
                                 init_method = op_info["init_method"],
                                 prev_name=op_info["prev_name"])
    elif "t5-post-process" in op_info["name"]:
        op = T5PostProcess(name=op_info["name"], 
                                 op_index = op_info["op_index"], 
                                 parallel_output = op_info["parallel_output"],
                                 init_method = op_info["init_method"],
                                 prev_name=op_info["prev_name"])
    else:
        raise RuntimeError(f"operator {op_info['name']} is not supported.")
    return op


# ================ FlexPipe operators definition ================ 

## Operators for Transformers

class Embedding(RopalaModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, op_index,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes,
                 name=None, prev_name=None):
        super(Embedding, self).__init__(op_index, name, prev_name)
        args = get_args()
        self.hidden_size = args.hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
    
        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            args.padded_vocab_size, self.hidden_size,
            init_method=self.init_method)
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(
            num_embeddings=args.max_position_embeddings, embedding_dim =self.hidden_size)
        self._position_embeddings_key = 'position_embeddings'
        # Initialize the position embeddings.
        self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                        self.hidden_size)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.fp32_residual_connection = args.fp32_residual_connection

        ## tensor shape information
        if self.name == "encoder-embedding":
            # self.input_tensors_info = {"input_tensor": {"shape": [], "tp_split_dim": -1, "dp_split_dim": -1}}
            self.input_tensors_info = {"enc_input_ids": {"shape": [args.micro_batch_size, args.seq_length], "tp_split_dim": -1, "dp_split_dim": -1},
                                        "enc_position_ids": {"shape": [args.micro_batch_size, args.seq_length], "tp_split_dim": -1, "dp_split_dim": -1}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}}
            if args.model_name == "gpt":
                self.shared_weights_info = {"word_embeddings": {"root": True, "sharing_with_ops": [13*args.num_layers +2], "shape": [args.padded_vocab_size, args.hidden_size], "tp_split_dim": 0, "dp_split_dim": -1}}
            elif args.model_name == "t5":
                self.shared_weights_info = {"word_embeddings": {"root": True, "sharing_with_ops": [args.num_layers*13+2, args.num_layers*(13+21)+4], "shape": [args.padded_vocab_size, args.hidden_size], "tp_split_dim": 0, "dp_split_dim": -1},
                                            "position_embeddings": {"root": True, "sharing_with_ops": [args.num_layers*13+2], "shape": [args.max_position_embeddings, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": -1}}
        elif self.name == "decoder-embedding":
            self.input_tensors_info = {"encoder_output": {"shape": [args.micro_batch_size, args.seq_length, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1},
                                        "encoder_output": {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}}
            self.shared_weights_info ={"word_embeddings": {"root": False, "sharing_with_ops": [0], "shape": [args.padded_vocab_size, args.hidden_size], "tp_split_dim": 0, "dp_split_dim": -1},
                                        "position_embeddings": {"root": False, "sharing_with_ops": [0], "shape": [args.max_position_embeddings, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": -1}}
        ## tensor resharding information
        if self.name == "encoder-embedding":
            self.required_input_specs = {"enc_input_ids": {}, "enc_position_ids": {}}
            self.output_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}}    
            self.output_mats_info = {"hidden_states": {"from": "enc_input_ids", "trans":[4, 1, 2, 3, 0]}}  
        elif self.name == "decoder-embedding":
            self.required_input_specs = {"encoder_output": {}}
            self.output_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]},
                                    "encoder_output": {}}    
            self.output_mats_info = {"hidden_states": {"from": "encoder_output", "trans":[4, 1, 2, 3, 0]},
                                    "encoder_output": {}}  

        ## profiling information
        self.weight_size = (args.padded_vocab_size * args.hidden_size)/self.tp_size + args.max_position_embeddings * args.hidden_size

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        args = get_args()
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}
        args = get_args()

        if self.name == "encoder-embedding":
            input_ids = input_tensors["enc_input_ids"]
            position_ids = input_tensors["enc_position_ids"]
        elif self.name == "decoder-embedding":
            # input_ids = input_tensors["dec_input_ids"]
            # position_ids = input_tensors["dec_position_ids"]   
            def t5_position_ids(token_ids):
                # Create position ids
                seq_length = token_ids.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long,
                                            device=token_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

                return position_ids
            input_ids = torch.rand((args.micro_batch_size//self.dp_size, args.decoder_seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * args.padded_vocab_size                    
            position_ids = t5_position_ids(input_ids)

        tokentype_ids=None

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        # transpose
        if self.fp32_residual_connection:
            embeddings = embeddings.transpose(0, 1).contiguous().float()
        # Otherwise, leave it as is.
        else:
            embeddings = embeddings.transpose(0, 1).contiguous()

        output_tensors["hidden_states"] = embeddings
        if self.name == "decoder-embedding":
            output_tensors["encoder_output"] = input_tensors["encoder_output"].transpose(0, 1).contiguous()

        return output_tensors

class ParallelLayerNormOp(RopalaModule):
    def __init__(self, op_index, name=None, prev_name=None):
        super(ParallelLayerNormOp, self).__init__(op_index, name, prev_name)
        args = get_args()
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if self.name in ["enc-1st-layernorm", "dec-1st-layernorm"]:
            extra_tensor_name = "hidden_states_1"
            extra_tensor_send_to = 7
        elif self.name == "enc-2nd-layernorm":
            extra_tensor_name = "hidden_states_2"
            extra_tensor_send_to = 4
        elif self.name == "dec-2nd-layernorm":
            extra_tensor_name = "hidden_states_2"
            extra_tensor_send_to = 7    
        elif self.name == "dec-3rd-layernorm":
            extra_tensor_name = "hidden_states_3"
            extra_tensor_send_to = 4                        
        else:
            extra_tensor_name = None

        # if "enc" in self.name:
        hidden_states_size = [args.seq_length, args.micro_batch_size, args.hidden_size]
        if "dec" in self.name: 
            hidden_states_size = [args.decoder_seq_length, args.micro_batch_size, args.hidden_size]
            encoder_output_size = [args.seq_length, args.micro_batch_size, args.hidden_size]

        ## tensor shape information
        self.input_tensors_info = {"hidden_states": {"shape": list(hidden_states_size), "tp_split_dim": -1, "dp_split_dim": 1}}
        self.output_tensors_info = {"hidden_states": {"shape": list(hidden_states_size), "tp_split_dim": -1, "dp_split_dim": 1}}
        
        if args.model_name == "gpt" and self.name == "final-layernorm":
            self.output_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.seq_length, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 0}}
        elif args.model_name == "t5" and self.name == "enc-final-layernorm":
            self.output_tensors_info = {"encoder_output": {"shape": [args.micro_batch_size, args.seq_length, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 0}}

        if "dec" in self.name:
            self.input_tensors_info["encoder_output"] = {"shape": list(encoder_output_size), "tp_split_dim": -1, "dp_split_dim": 1}
            self.output_tensors_info["encoder_output"] = {"shape": list(encoder_output_size), "tp_split_dim": -1, "dp_split_dim": 1}

        if extra_tensor_name is not None:
            self.output_extra_tensors_info = {extra_tensor_name: {"shape": list(hidden_states_size), "tp_split_dim": -1, "dp_split_dim": 1, "send_to": extra_tensor_send_to}}
        ## tensor resharding information
        self.required_input_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]} }
        if args.model_name == "gpt" and self.name == "final-layernorm":
            self.output_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]}} 
            self.output_mats_info = {"hidden_states": {"from": "hidden_states", "trans":[0, 1, 3, 2, 4]}}   
        elif args.model_name == "t5" and self.name == "enc-final-layernorm":
            self.output_specs = {"encoder_output": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]}} 
            self.output_mats_info = {"encoder_output": {"from": "hidden_states", "trans":[0, 1, 3, 2, 4]}}              
        else:
            self.output_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}} 
            self.output_mats_info = {"hidden_states": {"from": "hidden_states", "trans":[0, 1, 2, 3, 4]}}
        if extra_tensor_name is not None:
            self.output_extra_specs = {extra_tensor_name: {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}} 
            self.output_extra_mats_info = {extra_tensor_name: {"from": "hidden_states", "trans":[0, 1, 2, 3, 4]}}

        if "dec" in self.name:
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}

        if "final-layernorm" in self.name:
            input_tensors["hidden_states"] = input_tensors["hidden_states"].transpose(0, 1).contiguous()
        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]

        for key in sorted(self.output_extra_tensors_info):
            if self.output_extra_tensors_info[key]["cross_stage"]:
                output_extra_tensors[key] = input_tensors["hidden_states"]
                input_extra_tensors[key] = input_tensors["hidden_states"] 
            else:
                input_extra_tensors[key] = input_tensors["hidden_states"]     

            if "1st" in self.name: 
                self.new_input_extra_tensors = {"hidden_states_1"}            
            elif "2nd" in self.name:
                self.new_input_extra_tensors = {"hidden_states_2"}     

        layernorm_output = self.input_layernorm(input_tensors["hidden_states"])

        if self.name == "enc-final-layernorm": # this is for T5 model, the last layernorm of encoders.
            output_tensors["encoder_output"] = layernorm_output
        else:
            output_tensors["hidden_states"] = layernorm_output

        return output_tensors

class ParallelQKVOp(RopalaModule):
    def __init__(self, init_method,
                 attention_type, op_index, algo=0, name=None, prev_name=None):
        super(ParallelQKVOp, self).__init__(op_index, name, prev_name)
        args = get_args()

        projection_size = args.kv_channels * args.num_attention_heads
        self.attention_type = attention_type
        if algo == 0:
            self.algo = "column"
        elif algo == 1:
            self.algo = "row"
        else:
            raise RuntimeError("algo not implemented.")

        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        if args.resharding_stages[rank_in_pipeline]:
            self.resharding = True
            if attention_type == AttnType.self_attn:
                if self.algo == "column":
                    self.query_key_value = mpu.NewColumnParallelLinear(
                        args.hidden_size,
                        3 * projection_size,
                        self.tp_size,
                        gather_output=False,
                        init_method=init_method)
                elif self.algo == "row":
                    self.query_key_value = mpu.NewRowParallelLinear(
                        args.hidden_size,
                        3 * projection_size,
                        self.tp_size,
                        init_method=init_method)            
            else:
                raise RuntimeError("flexible GEMM partition is not implemented for T5 yet.")
                assert attention_type == AttnType.cross_attn
                if self.algo == "column":
                    self.query = mpu.NewColumnParallelLinear(
                        args.hidden_size,
                        projection_size,
                        self.tp_size,
                        gather_output=False,
                        init_method=init_method)
                elif self.algo == "row":
                    self.query = mpu.NewRowParallelLinear(
                        args.hidden_size,
                        projection_size,
                        self.tp_size,
                        init_method=init_method,
                        skip_bias_add=False)     

                if self.algo == "column":
                    self.key_value = mpu.NewColumnParallelLinear(
                        args.hidden_size,
                        2 * projection_size,
                        self.tp_size,
                        gather_output=False,
                        init_method=init_method)
                elif self.algo == "row":
                    self.key_value = mpu.NewRowParallelLinear(
                        args.hidden_size,
                        2 * projection_size,
                        self.tp_size,
                        init_method=init_method,
                        skip_bias_add=True)                                
            world_size = self.tp_size
        else:
            self.resharding = False
            # Strided linear layer.
            if attention_type == AttnType.self_attn:
                self.query_key_value = mpu.ColumnParallelLinear(
                    args.hidden_size,
                    3 * projection_size,
                    gather_output=False,
                    init_method=init_method)
            else:
                assert attention_type == AttnType.cross_attn
                self.query = mpu.ColumnParallelLinear(
                    args.hidden_size,
                    projection_size,
                    gather_output=False,
                    init_method=init_method)

                self.key_value = mpu.ColumnParallelLinear(
                    args.hidden_size,
                    2 * projection_size,
                    gather_output=False,
                    init_method=init_method)
            world_size = mpu.get_tensor_model_parallel_world_size()

        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)

        if self.algo == "row":
            assert self.resharding is True        
        ## shape information
        if self.name == "enc-attention-qkv":
            extra_tensor_name = "value_layer"
            self.input_tensors_info = {"hidden_states": {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}}
            self.output_tensors_info = {"query_layer": {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1},
                                        "key_layer": {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1}}
            self.output_extra_tensors_info = {extra_tensor_name: {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1, "send_to": 4}} 
        elif self.name == "dec-attention-qkv-1":
            extra_tensor_name = "value_layer_1"
            self.input_tensors_info = {"hidden_states": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1},
                                        "encoder_output": {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}}
            self.output_tensors_info = {
                "query_layer": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1},
                "key_layer": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1},
                "encoder_output": {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            }
            self.output_extra_tensors_info = {
                extra_tensor_name: {"shape": [args.decoder_seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1, "send_to": 4}
            }  
        elif self.name == "dec-attention-qkv-2":
            extra_tensor_name = "value_layer_2"
            self.input_tensors_info = {
                "hidden_states": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1},
                "encoder_output": {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            }
            self.output_tensors_info = {
                "query_layer": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1},
                "key_layer": {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1},
                "encoder_output": {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            }
            self.output_extra_tensors_info = {
                extra_tensor_name: {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1, "send_to": 4}
            }
        ## resharding information
        if self.algo == "column":
            self.required_input_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]} }      
            self.output_specs = {"query_layer": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]},
                                "key_layer": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}}     
            self.output_mats_info = {"query_layer": {"from": "hidden_states", "trans":[4, 1, 2, 3, 0]},
                                    "key_layer": {"from": "hidden_states", "trans":[4, 1, 2, 3, 0]}}    
            self.output_extra_specs = {extra_tensor_name: {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}} 
            self.output_extra_mats_info = {extra_tensor_name: {"from": "hidden_states", "trans":[4, 1, 2, 3, 0]}}
        elif self.algo == "row":
            self.required_input_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}}      
            self.output_specs = {"query_layer": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]},
                                "key_layer": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}}     
            self.output_mats_info = {"query_layer": {"from": "hidden_states", "trans":[4, 1, 2, 3, 0]},
                                    "key_layer": {"from": "hidden_states", "trans":[4, 1, 2, 3, 0]}} 
            self.output_extra_specs = {extra_tensor_name: {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}} 
            self.output_extra_mats_info = {extra_tensor_name: {"from": "hidden_states", "trans":[4, 1, 2, 3, 0]}}
        if self.name =="dec-attention-qkv-1":
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}    
        elif  self.name =="dec-attention-qkv-2":     
            if self.algo == "column":  
                self.required_input_specs["encoder_output"] = {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}
                self.output_specs["encoder_output"] = {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}
                self.output_mats_info["encoder_output"] = {"from": "encoder_output", "trans":[0, 1, 2, 3, 4]}
            elif self.algo == "row":  
                self.required_input_specs["encoder_output"] = {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}
                self.output_specs["encoder_output"] = {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}
                self.output_mats_info["encoder_output"] = {"from": "encoder_output", "trans":[0, 1, 2, 3, 4]}                
        ## profiling
        self.weight_size = args.hidden_size * projection_size * 3 / self.tp_size     

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):

        output_tensors = {}        
        hidden_states = input_tensors["hidden_states"]
        if self.resharding and self.algo == "column":
            hidden_states = new_copy_to_tensor_model_parallel_region(self.op_index, hidden_states, self.required_input_specs["hidden_states"], self.input_mats["hidden_states"])
            
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            if self.resharding and self.algo == "row":
                output_mats = self.input_mats["hidden_states"].transpose([4, 1, 2, 3, 0])
                mixed_x_layer = new_reduce_from_tensor_model_parallel_region(self.op_index, mixed_x_layer, self.output_specs["query_layer"], output_mats)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            if self.algo == "column":
                new_tensor_shape = mixed_x_layer.size()[:-1] + \
                    (self.num_attention_heads_per_partition,
                    3 * self.hidden_size_per_attention_head)
            elif self.algo == "row":
                new_tensor_shape = mixed_x_layer.size()[:-1] + \
                    (self.num_attention_heads_per_partition * self.tp_size,
                    3 * self.hidden_size_per_attention_head)                
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3, contiguous_split_chunks=True)
        else:
            
            encoder_output = input_tensors["encoder_output"]
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)
            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2, contiguous_split_chunks=True)
            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        output_tensors["query_layer"] = query_layer
        output_tensors["key_layer"] = key_layer

        for key in sorted(self.output_extra_tensors_info):
            if self.output_extra_tensors_info[key]["cross_stage"]:
                output_extra_tensors[key] = value_layer
                input_extra_tensors[key] = value_layer
            else:
                input_extra_tensors[key] = value_layer
        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]

        if self.resharding:
            self.new_input_extra_tensors = {"value_layer"}    

        return output_tensors

class ParallelAttentionScoreOp(RopalaModule):
    def __init__(self, layer_number, op_index, name=None, prev_name=None):
        super(ParallelAttentionScoreOp, self).__init__(op_index, name, prev_name)
        args = get_args()

        projection_size = args.kv_channels * args.num_attention_heads
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        ## shape information
        if self.name == "enc-attention-score":
            self.input_tensors_info = {"query_layer": {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1},
                                        "key_layer": {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
        elif self.name == "dec-attention-score-1":
            self.input_tensors_info = {
                "query_layer": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1},
                "key_layer": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1}
            }
            self.output_tensors_info = {
                "hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.decoder_seq_length], "tp_split_dim": 1, "dp_split_dim": 0}
            }
        elif self.name == "dec-attention-score-2":
            self.input_tensors_info = {
                "query_layer": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1},
                "key_layer": {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1}
            }
            self.output_tensors_info = {
                "hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}
            }
        ## resharding information
        self.required_input_specs = {"query_layer": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]},
                                        "key_layer": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}}   
        self.output_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [self.dp_size, self.tp_size, 1]}} 
        self.output_mats_info = {"hidden_states": {"from": "query_layer", "trans":[0, 1, 3, 4, 2]}}  

        if "dec" in self.name:
            self.input_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.output_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}        
        query_layer = input_tensors["query_layer"]
        key_layer = input_tensors["key_layer"]
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        output_tensors["hidden_states"] = attention_scores

        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]        

        return output_tensors

class ParallelSoftmaxFusionOp(RopalaModule):
    def __init__(self, layer_number, attn_mask_type, op_index, name=None, prev_name=None):
        super(ParallelSoftmaxFusionOp, self).__init__(op_index, name, prev_name)
        args = get_args()   

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True        
        self.fp16 = args.fp16
        self.bf16 = args.bf16
        self.layer_number = max(1, layer_number)

        coeff = None
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        self.resharding = args.resharding_stages[rank_in_pipeline]

        ## shape info
        if self.name == "enc-attention-softmax":
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.input_extra_tensors_info = {"enc_attention_mask": {"shape": [args.micro_batch_size, 1, args.seq_length, args.seq_length], "tp_split_dim": -1, "dp_split_dim": -1, "recv_from": 0}}
            self.required_input_extra_specs = {"enc_attention_mask": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]}}
        elif self.name == "dec-attention-softmax-1":
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.decoder_seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.decoder_seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.input_extra_tensors_info = {"dec_attention_mask": {"shape": [args.micro_batch_size, 1, args.decoder_seq_length, args.decoder_seq_length], "tp_split_dim": -1, "dp_split_dim": -1, "recv_from": 0}}
            self.required_input_extra_specs = {"dec_attention_mask": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]}}
        elif self.name == "dec-attention-softmax-2":
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.input_extra_tensors_info = {"enc_dec_attention_mask": {"shape": [args.micro_batch_size, 1, args.decoder_seq_length, args.encoder_seq_length], "tp_split_dim": -1, "dp_split_dim": -1, "recv_from": 0}}
            self.required_input_extra_specs = {"enc_dec_attention_mask": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]}}
        ## resharding info
        self.required_input_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [self.dp_size, self.tp_size, 1]}}  
        self.output_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [self.dp_size, self.tp_size, 1]}} 
        self.output_mats_info = {"hidden_states": {"from": "hidden_states", "trans":[0, 1, 2, 3, 4]}}  

        if "dec" in self.name:
            self.input_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.output_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        args = get_args()
        timers = get_timers()
        output_tensors = {}                  
        hidden_states = input_tensors["hidden_states"]
        micro_batch_size = args.micro_batch_size // self.dp_size
        if self.name == "enc-attention-softmax":          
            attention_mask = input_extra_tensors["enc_attention_mask"][0:micro_batch_size, :, :, :]
        elif self.name == "dec-attention-softmax-1":
            attention_mask = input_extra_tensors["dec_attention_mask"][0:micro_batch_size, :, :, :]  
        elif self.name == "dec-attention-softmax-2":
            attention_mask = input_extra_tensors["enc_dec_attention_mask"][0:micro_batch_size, :, :, :]    
        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(hidden_states, attention_mask)
        output_tensors["hidden_states"] = attention_probs

        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]       

        return output_tensors

class ParallelSoftmaxDropoutOp(RopalaModule):
    def __init__(self, op_index, name=None, prev_name=None):
        super(ParallelSoftmaxDropoutOp, self).__init__(op_index, name, prev_name)
        args = get_args() 
        self.attention_dropout = torch.nn.Dropout(p=args.attention_dropout, inplace=False)

        ## shape info
        if self.name == "enc-attention-dropout":
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
        elif self.name == "dec-attention-dropout-1":
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.decoder_seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.decoder_seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}   
        elif self.name == "dec-attention-dropout-2":
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
        ## resharding info
        self.required_input_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [self.dp_size, self.tp_size, 1]}}  
        self.output_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [self.dp_size, self.tp_size, 1]}} 
        self.output_mats_info = {"hidden_states": {"from": "hidden_states", "trans":[0, 1, 2, 3, 4]}}  
        if "dec" in self.name:
            self.input_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.output_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}       
        hidden_states = input_tensors["hidden_states"]
        with mpu.get_cuda_rng_tracker().fork():    
            attention_probs = self.attention_dropout(hidden_states)

        output_tensors["hidden_states"] = attention_probs

        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]    

        return output_tensors

class ParallelContextOp(RopalaModule):
    def __init__(self, op_index, name=None, prev_name=None):
        super(ParallelContextOp, self).__init__(op_index, name, prev_name)
        args = get_args()
        projection_size = args.kv_channels * args.num_attention_heads
        self.hidden_size_per_partition = mpu.divide(projection_size, self.tp_size)   

        ## shape info
        if self.name == "enc-attention-context":
            extra_tensor_name = "value_layer"
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.seq_length, args.micro_batch_size, args.kv_channels * args.num_attention_heads], "tp_split_dim": 2, "dp_split_dim": 1}}
            self.input_extra_tensors_info = {extra_tensor_name: {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1, "recv_from": -4}}   
        elif self.name == "dec-attention-context-1":
            extra_tensor_name = "value_layer_1"
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.decoder_seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.kv_channels * args.num_attention_heads], "tp_split_dim": 2, "dp_split_dim": 1}}
            self.input_extra_tensors_info = {extra_tensor_name: {"shape": [args.decoder_seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1, "recv_from": -4}}    
        elif self.name == "dec-attention-context-2":
            extra_tensor_name = "value_layer_2"
            self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.num_attention_heads, args.decoder_seq_length, args.seq_length], "tp_split_dim": 1, "dp_split_dim": 0}}
            self.output_tensors_info = {"hidden_states": {"shape": [args.decoder_seq_length, args.micro_batch_size, args.kv_channels * args.num_attention_heads], "tp_split_dim": 2, "dp_split_dim": 1}}
            self.input_extra_tensors_info = {extra_tensor_name: {"shape": [args.seq_length, args.micro_batch_size, args.num_attention_heads, args.kv_channels], "tp_split_dim": 2, "dp_split_dim": 1, "recv_from": -4}}                           
        ## resharding info
        self.required_input_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [self.dp_size, self.tp_size, 1]}}  
        self.output_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}} 
        self.output_mats_info = {"hidden_states": {"from": "hidden_states", "trans":[0, 1, 4, 2, 3]}}  
        self.required_input_extra_specs = {extra_tensor_name: {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}}
        if "dec" in self.name:
            self.input_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.output_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False): 
        output_tensors = {}    

        if profiling:
            if self.name == "enc-attention-context":
                value_layer = input_extra_tensors["value_layer"]
            elif self.name == "dec-attention-context-1":
                value_layer = input_extra_tensors["value_layer_1"]
            elif self.name == "dec-attention-context-2":     
                value_layer = input_extra_tensors["value_layer_2"]
        else:
            if self.name == "enc-attention-context":
                value_layer = input_extra_tensors.pop("value_layer")
            elif self.name == "dec-attention-context-1":
                value_layer = input_extra_tensors.pop("value_layer_1")
            elif self.name == "dec-attention-context-2":     
                value_layer = input_extra_tensors.pop("value_layer_2")


        attention_probs = input_tensors["hidden_states"]
        # =========================
        # Context layer. [sq, b, hp]
        # =========================
        seq_len = attention_probs.size()[-2]
        # [sk, b, np, hn] --> [b, np, sq, hn]
        # may need modification for T5 model
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       seq_len, #query_layer.size(0),
                       value_layer.size(3))        
        # change view [sk, b * np, hn] 
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0,1))
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output_tensors["hidden_states"] = context_layer

        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]    

        return output_tensors

"""
AttentionDense: dim1 = args.kv_channels * args.num_attention_heads, dim2 = args.hidden_size
MLP-GEMM-1: dim1 = args.hidden_size, dim2 = args.ffn_hidden_size
MLP-GEMM-2: dim1 = args.ffn_hidden_size, dim2 = args.hidden_size
"""
class ParallelGEMM(RopalaModule):
    def __init__(self, op_index, init_method, dim1_size, dim2_size, algo=0, name=None, prev_name=None):
        super(ParallelGEMM, self).__init__(op_index, name, prev_name)
        args = get_args()

        if self.name in ["enc-attention-dense", "enc-MLP-GEMM-2", "dec-attention-dense-1", "dec-attention-dense-2", "dec-MLP-GEMM-2"]:
            if algo == 0:
                self.algo = "row"
            elif algo == 1:
                self.algo = "column"
            else:
                raise RuntimeError("algo not implemented.")
        elif self.name in ["enc-MLP-GEMM-1", "dec-MLP-GEMM-1"]:
            if algo == 0:
                self.algo = "column"
            elif algo == 1:
                self.algo = "row"
            else:
                raise RuntimeError("algo not implemented.")         
        else:
            raise RuntimeError(f"op {self.name} not implemented.")   

        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        if args.resharding_stages[rank_in_pipeline]:
            self.resharding = True
            if self.algo == "row":
                self.dense = mpu.NewRowParallelLinear(dim1_size, dim2_size, self.tp_size, init_method=init_method, skip_bias_add=True)       
            elif self.algo == "column": 
                self.dense = mpu.NewColumnParallelLinear(dim1_size, dim2_size, self.tp_size, gather_output=False, init_method=init_method, skip_bias_add=True)       
        else:
            self.resharding = False
            if self.algo == "row":
                self.dense = mpu.RowParallelLinear(dim1_size, dim2_size, input_is_parallel=True, init_method=init_method, skip_bias_add=True)
            elif self.algo == "column":
                self.dense = mpu.ColumnParallelLinear(dim1_size, dim2_size, gather_output=False, init_method=init_method, skip_bias_add=True)  

        if "enc" in self.name:
            seq_len = args.seq_length
        elif "dec" in self.name:
            seq_len = args.decoder_seq_length
        else:
            raise RuntimeError("check the op config.")

        if self.algo == "row":
            ## shape information
            self.input_tensors_info = {"hidden_states": {"shape": [seq_len, args.micro_batch_size, dim1_size], "tp_split_dim": 2, "dp_split_dim": 1}}
            self.output_tensors_info = {"output_tensor": {"shape": [seq_len, args.micro_batch_size, dim2_size], "tp_split_dim": -1, "dp_split_dim": 1}}
            self.output_extra_tensors_info = {"output_bias": {"shape": [dim2_size], "tp_split_dim": -1, "dp_split_dim": -1, "send_to": 1}}  
            ## resharding information
            self.required_input_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}} 
            self.output_specs = {"output_tensor": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}}        
            self.output_mats_info = {"output_tensor": {"from": "hidden_states", "trans":[4, 1, 2, 3, 0]}} 
            self.output_extra_specs = {"output_bias": {"R": self.tp_size * self.dp_size, "V": 1, "dims": [1, 1, 1]}} 
            self.output_extra_mats_info = {"output_bias": {"from": "hidden_states", "trans":[0, 1, 2, 3, 4]}}
        elif self.algo == "column":
            ## shape information
            self.input_tensors_info = {"hidden_states": {"shape": [seq_len, args.micro_batch_size, dim1_size], "tp_split_dim": -1, "dp_split_dim": 1}}
            self.output_tensors_info = {"output_tensor": {"shape": [seq_len, args.micro_batch_size, dim2_size], "tp_split_dim": 2, "dp_split_dim": 1}}
            self.output_extra_tensors_info = {"output_bias": {"shape": [dim2_size], "tp_split_dim": 0, "dp_split_dim": -1, "send_to": 1}}  
            ## resharding information            
            self.required_input_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]} }      
            self.output_specs = {"output_tensor": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}}     
            self.output_mats_info = {"output_tensor": {"from": "hidden_states", "trans":[4, 1, 2, 3, 0]}}     
            self.output_extra_specs = {"output_bias": {"R": self.dp_size, "V": 1, "dims": [self.tp_size, 1, 1]}} 
            self.output_extra_mats_info = {"output_bias": {"from": "hidden_states", "trans":[3, 1, 0, 2, 4]}}
        if "dec" in self.name:
            self.input_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.output_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}
        ## profiling
        self.weight_size = dim1_size * dim2_size / self.tp_size

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}             
        hidden_states = input_tensors["hidden_states"]

        if self.resharding and self.algo == "column":      
            hidden_states = new_copy_to_tensor_model_parallel_region(self.op_index, hidden_states, self.required_input_specs["hidden_states"], self.input_mats["hidden_states"])

        output, bias = self.dense(hidden_states)

        if self.resharding and self.algo == "row":
            output_mats = self.input_mats["hidden_states"].transpose([4, 1, 2, 3, 0])
            output = new_reduce_from_tensor_model_parallel_region(self.op_index, output, self.output_specs["output_tensor"], output_mats)

        output_tensors["output_tensor"] = output


        if self.output_extra_tensors_info["output_bias"]["cross_stage"]:
            output_extra_tensors["output_bias"] = bias
            input_extra_tensors["output_bias"] = bias
        else:
            input_extra_tensors["output_bias"] = bias

        if self.resharding:
            self.new_input_extra_tensors = {"output_bias"}       

        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]    

        return output_tensors

class ParallelDropoutOp(RopalaModule):
    def __init__(self, op_index, name=None, prev_name=None):
        super(ParallelDropoutOp, self).__init__(op_index, name, prev_name)
        args = get_args()
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.hidden_dropout = args.hidden_dropout

        if self.name == "enc-post-attention-dropout":
            input_extra_name = ["hidden_states_1", "output_bias"]
            recv_from = -7
        elif self.name == "enc-post-MLP-dropout":
            input_extra_name = ["hidden_states_2", "output_bias"]
            recv_from = -4
        elif self.name == "dec-post-attention-dropout-1":
            input_extra_name = ["hidden_states_1", "output_bias"]
            recv_from = -7
        elif self.name == "dec-post-attention-dropout-2":
            input_extra_name = ["hidden_states_2", "output_bias"]
            recv_from = -7     
        elif self.name == "dec-post-MLP-dropout":
            input_extra_name = ["hidden_states_3", "output_bias"]
            recv_from = -4
        else:
            input_extra_name = None

        if "enc" in self.name:
            seq_len = args.seq_length
        elif "dec" in self.name:
            seq_len = args.decoder_seq_length
        else:
            raise RuntimeError("check the op config.")
        
        ## shape info
        self.input_tensors_info = {"output_tensor": {"shape": [seq_len, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}}
        self.output_tensors_info = {"hidden_states": {"shape": [seq_len, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}}
        if input_extra_name is not None:
            self.input_extra_tensors_info = {input_extra_name[0]: {"shape": [seq_len, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1, "recv_from": recv_from},
                                             input_extra_name[1]: {"shape": [args.hidden_size], "tp_split_dim": -1, "dp_split_dim": -1, "recv_from": -1}}  

        ## resharding info
        self.required_input_specs = {"output_tensor": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}} 
        self.output_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]}} 
        self.output_mats_info = {"hidden_states": {"from": "output_tensor", "trans":[0, 1, 2, 3, 4]}}   
        if input_extra_name is not None:
            self.required_input_extra_specs = {input_extra_name[0]: {"R": self.tp_size, "V": 1, "dims": [1, self.dp_size, 1]},
                                               input_extra_name[1]: {"R": self.tp_size * self.dp_size, "V": 1, "dims": [1, 1, 1]}} 
        if "dec" in self.name:
            self.input_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.output_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):   
        output_tensors = {}

        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        if "enc-post-attention-dropout" in self.name or self.name ==  "dec-post-attention-dropout-1":
            with torch.enable_grad():       
                layernorm_input = bias_dropout_add_func(
                    input_tensors["output_tensor"],
                    input_extra_tensors["output_bias"].expand_as(input_extra_tensors["hidden_states_1"]),
                    input_extra_tensors["hidden_states_1"],
                    self.hidden_dropout) 
            if not profiling:
                input_extra_tensors.pop("output_bias")
                input_extra_tensors.pop("hidden_states_1")

        elif self.name == "enc-post-MLP-dropout" or self.name == "dec-post-attention-dropout-2":
            with torch.enable_grad():       
                layernorm_input = bias_dropout_add_func(
                    input_tensors["output_tensor"],
                    input_extra_tensors["output_bias"].expand_as(input_extra_tensors["hidden_states_2"]),
                    input_extra_tensors["hidden_states_2"],
                    self.hidden_dropout)   
            if not profiling:
                input_extra_tensors.pop("output_bias")
                input_extra_tensors.pop("hidden_states_2")

        elif self.name == "dec-post-MLP-dropout" :
            with torch.enable_grad():       
                layernorm_input = bias_dropout_add_func(
                    input_tensors["output_tensor"],
                    input_extra_tensors["output_bias"].expand_as(input_extra_tensors["hidden_states_3"]),
                    input_extra_tensors["hidden_states_3"],
                    self.hidden_dropout)                
            if not profiling:
                input_extra_tensors.pop("output_bias")
                input_extra_tensors.pop("hidden_states_3")

        output_tensors["hidden_states"] = layernorm_input
        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]    

        return output_tensors

class ParallelMlpGeLUOp(RopalaModule):
    def __init__(self, op_index, name=None, prev_name=None):
        super(ParallelMlpGeLUOp, self).__init__(op_index, name, prev_name)
        args = get_args()
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu

        self.resharding = True
        self.elementwise = True      

        if "enc" in self.name:
            seq_len = args.seq_length
        elif "dec" in self.name:
            seq_len = args.decoder_seq_length
        else:
            raise RuntimeError("check the op config.")

        ## shape info
        self.input_tensors_info = {"output_tensor": {"shape": [seq_len, args.micro_batch_size, args.ffn_hidden_size], "tp_split_dim": 2, "dp_split_dim": 1}}
        self.output_tensors_info = {"hidden_states": {"shape": [seq_len, args.micro_batch_size, args.ffn_hidden_size], "tp_split_dim": 2, "dp_split_dim": 1}}
        self.input_extra_tensors_info = {"output_bias": {"shape": [args.ffn_hidden_size], "tp_split_dim": 0, "dp_split_dim": -1, "recv_from": -1}}  
        ## resharding info                     
        self.required_input_specs = {"output_tensor": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}}      
        self.output_specs = {"hidden_states": {"R": 1, "V": 1, "dims": [1, self.dp_size, self.tp_size]}}     
        self.output_mats_info = {"hidden_states": {"from": "output_tensor", "trans":[0, 1, 2, 3, 4]}}    
        self.required_input_extra_specs = {"output_bias": {"R": self.dp_size, "V": 1, "dims": [self.tp_size, 1, 1]}} 
        if "dec" in self.name:
            self.input_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.output_tensors_info["encoder_output"] = {"shape": [args.seq_length, args.micro_batch_size, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 1}
            self.required_input_specs["encoder_output"] = {}
            self.output_specs["encoder_output"] = {}
            self.output_mats_info["encoder_output"] = {}

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):           
        output_tensors = {}
        intermediate_parallel = input_tensors["output_tensor"]

        if profiling:
            bias_parallel = input_extra_tensors["output_bias"]
        else:
            bias_parallel = input_extra_tensors.pop("output_bias")

        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)     
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)  

        output_tensors["hidden_states"] = intermediate_parallel

        if "dec" in self.name:
            output_tensors["encoder_output"] = input_tensors["encoder_output"]   

        return output_tensors

class GPTPostProcess(RopalaModule):
    def __init__(self, parallel_output, init_method, op_index, name=None, prev_name=None):
        super(GPTPostProcess, self).__init__(op_index, name, prev_name)
        args = get_args()

        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        if args.resharding_stages[rank_in_pipeline]:
            self.resharding = True       
            all_ranks = mpu.get_ranks_via_pipeline_stage(mpu.get_pipeline_model_parallel_rank())
            all_ranks_ = np.array(all_ranks).reshape(-1, self.tp_size)
            rank = torch.distributed.get_rank()
            for ranks in all_ranks_:
                if rank in ranks:
                    self.tp_group_ranks = ranks
            self.loss_func = mpu.new_vocab_parallel_cross_entropy
            self.lm_logits_func = new_parallel_lm_logits
        else:
            self.resharding = False
            self.loss_func = mpu.vocab_parallel_cross_entropy
            self.lm_logits_func = parallel_lm_logits

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        if self.resharding:
            self.word_embeddings = mpu.NewVocabParallelEmbedding(
                args.padded_vocab_size, args.hidden_size, self.tp_group_ranks,
                init_method=init_method)            
        else:
            self.word_embeddings = mpu.VocabParallelEmbedding(
                args.padded_vocab_size, args.hidden_size,
                init_method=init_method)
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True

        ## shape info
        self.input_tensors_info = {"hidden_states": {"shape": [args.micro_batch_size, args.seq_length, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 0}}            
        self.output_tensors_info = {"output_tensor": {"shape": [1], "tp_split_dim": -1, "dp_split_dim": -1}}
        self.shared_weights_info ={"word_embeddings": {"root": False, "sharing_with_ops": [0], "shape": [args.padded_vocab_size, args.hidden_size], "tp_split_dim": 0, "dp_split_dim": -1}}
        ## resharding info
        self.required_input_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }   
        self.output_specs = {"output_tensor": {}} 
        ## profiling
        self.weight_size = args.padded_vocab_size * args.hidden_size / self.tp_size

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        lm_output = input_tensors["hidden_states"]

        output_tensors = {}

        logit_weights = self.word_embeddings.weight
        # labels = input_extra_tensors["labels"]
        args = get_args()
        labels = torch.rand((args.micro_batch_size//self.dp_size, args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * args.padded_vocab_size

        if self.resharding:
            mpu.set_resharding_group(self.tp_group_ranks)
            mpu.set_resharding_dim(-1)
        output = self.lm_logits_func(
            lm_output,
            logit_weights,
            self.parallel_output)
        if labels is None:
            output_tensors["output_tensor"] = output
        else:
            if self.resharding:
                mpu.set_resharding_group(self.tp_group_ranks)
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = self.loss_func(output, labels)
            else:
                loss = self.loss_func(output.float(), labels)
            output_tensors["output"] = loss

        return output_tensors

class T5PostProcess(RopalaModule):
    def __init__(self, parallel_output, init_method, op_index, name=None, prev_name=None):
        super(T5PostProcess, self).__init__(op_index, name, prev_name)     
        args = get_args()
        self.word_embeddings = mpu.VocabParallelEmbedding(
            args.padded_vocab_size, args.hidden_size,
            init_method=init_method)
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        mpu_vocab_size = self.word_embeddings.weight.size(0)
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

        ## shape info
        self.input_tensors_info = {
                "hidden_states": {"shape": [args.micro_batch_size, args.decoder_seq_length, args.hidden_size], "tp_split_dim": -1, "dp_split_dim": 0}
            }
        self.output_tensors_info = {
                "output_tensor": {"shape": [1], "tp_split_dim": -1, "dp_split_dim": -1}
            }
        self.shared_weights_info ={
                "word_embeddings": {"root": False, "sharing_with_ops": [0], "shape": [args.padded_vocab_size, args.hidden_size], "tp_split_dim": 0, "dp_split_dim": -1}
            }
        ## resharding info
        self.required_input_specs = {"hidden_states": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }    
        self.output_specs = {"output_tensor": {}} 
        ## profiling
        self.weight_size = args.padded_vocab_size * args.hidden_size / self.tp_size

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False): 

        hidden_states = input_tensors["hidden_states"]
        if profiling:
            args = get_args()
            vocab_size = 30624
            lm_labels = torch.rand((args.micro_batch_size//self.dp_size, args.decoder_seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
        else:
            lm_labels = input_extra_tensors["labels"]
        output_tensors = {}

        lm_logits = parallel_lm_logits(hidden_states,
                                    self.word_embeddings.weight,
                                    self.parallel_output,
                                    bias=self.bias)
        
        if lm_labels is None:
            output_tensors["output_tensor"] = lm_logits
        else:
            if self.fp16_lm_cross_entropy:
                assert lm_logits.dtype == torch.half
                lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits, lm_labels)
            else:
                lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits.float(),
                                                           lm_labels)
            output_tensors["output"] = lm_loss   

        return output_tensors

## Operators for ResNet

class ParallelConvOp(RopalaModule):
    def __init__(self, 
        op_index: int, 
        in_channels: int, 
        input_width: int,        
        out_channels: int, 
        kernel_size: int = 1,         
        stride: int = 1,
        padding: int = 0,
        algo: int = 0,
        downsample: bool = False,
        residual_in_channels: int = 0,
        residual_input_width: int = 0,
        ## required information
        name: str = None,
        prev_name: str = None,
        send_residual: bool = False,
        recv_residual: bool = False,
        init_method=None,
    ):
        super(ParallelConvOp, self).__init__(op_index, name, prev_name)

        ## op-related information
        args = get_args()
        self.recv_residual = recv_residual
        self.send_residual = send_residual

        if self.recv_residual:
            self.input_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
            self.output_tensors_info = {"input": {"shape": [args.micro_batch_size, out_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
            self.input_extra_tensors_info = {"input_residual": {"shape": [args.micro_batch_size, residual_in_channels, residual_input_width, residual_input_width], "tp_split_dim": -1, "dp_split_dim": 0, "recv_from": -6}}       
        else:
            self.input_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
            self.output_tensors_info = {"input": {"shape": [args.micro_batch_size, out_channels, input_width//stride, input_width//stride], "tp_split_dim": -1, "dp_split_dim": 0}}

        if send_residual:
            self.output_extra_tensors_info = {"input_residual": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0, "send_to": 6}}

        if algo == 0:
            self.algo = "p_output"
        elif algo == 1:
            self.algo = "p_input"
        else:
            raise RuntimeError(f"algo {algo} for op {self.name} is not implemented.") 

        output_div_factor = 1
        input_div_factor = 1
        if self.algo == "p_output":
            output_div_factor = self.tp_size
        elif self.algo == "p_input":
            input_div_factor = self.tp_size

        if self.recv_residual:
            _in_channels = residual_in_channels
        else:
            _in_channels = in_channels

        if kernel_size > 0:
            self.conv = torch.nn.Conv2d(_in_channels//input_div_factor, out_channels//output_div_factor, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        else:
            self.conv = None

        self.weight_size = kernel_size * kernel_size * _in_channels * out_channels / (input_div_factor * output_div_factor)

        args = get_args()
        init_method = init_method_normal(args.init_method_std)

        if self.tp_size > 1 and self.conv is not None:
            if self.algo == "p_output":
                _initialize_affine_weight_gpu(self.conv.weight, init_method, partition_dim=0, stride=1)                
            elif self.algo == "p_input":
                _initialize_affine_weight_gpu(self.conv.weight, init_method, partition_dim=1, stride=1)  

        if downsample: 
            self.norm = torch.nn.BatchNorm2d(out_channels)

        self.required_input_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_mats_info = {"input": {"from": "input", "trans":[0, 1, 2, 3, 4]}} 
        if recv_residual:
            self.required_input_extra_specs = {"input_residual": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        if send_residual:
            self.output_extra_specs = {"input_residual": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
            self.output_extra_mats_info = {"input_residual": {"from": "input", "trans":[0, 1, 2, 3, 4]}} 

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}
        if self.recv_residual:
            if profiling:
                _input_residual = input_extra_tensors["input_residual"]
            else:
                _input_residual = input_extra_tensors.pop("input_residual")
            if self.conv is not None:
                if self.algo == "p_output":
                    input_residual = copy_to_tensor_model_parallel_region(_input_residual)
                elif self.algo == "p_input":
                    input_residual = prim_split(_input_residual, mpu.get_tensor_model_parallel_ranks_via_op_index(self.op_index), dim=1)
                residual = self.conv(input_residual)
                if self.algo == "p_output":
                    residual = prim_all_gather(residual, mpu.get_tensor_model_parallel_ranks_via_op_index(self.op_index), dim=1)
                elif self.algo == "p_input":
                    residual = prim_all_reduce(residual, mpu.get_tensor_model_parallel_ranks_via_op_index(self.op_index))
                residual = self.norm(residual)
            else:
                residual = _input_residual

            input_tensor = input_tensors["input"]
            output_tensor = input_tensor + residual
        else:
            if self.send_residual:
                if self.output_extra_tensors_info["input_residual"]["cross_stage"]:
                    output_extra_tensors["input_residual"] = input_tensors["input"]
                    input_extra_tensors["input_residual"] = input_tensors["input"]
                else:
                    input_extra_tensors["input_residual"] = input_tensors["input"]     
                self.new_input_extra_tensors = {"input_residual"}            

            if self.algo == "p_output":
                input_tensor = copy_to_tensor_model_parallel_region(input_tensors["input"])
            elif self.algo == "p_input":
                input_tensor = prim_split(input_tensors["input"], mpu.get_tensor_model_parallel_ranks_via_op_index(self.op_index), dim=1)

            output_tensor = self.conv(input_tensor)
            
            if self.algo == "p_output":
                output_tensor = prim_all_gather(output_tensor, mpu.get_tensor_model_parallel_ranks_via_op_index(self.op_index), dim=1)
            elif self.algo == "p_input":
                output_tensor = prim_all_reduce(output_tensor, mpu.get_tensor_model_parallel_ranks_via_op_index(self.op_index))

        output_tensors["input"] = output_tensor
        if DEBUG_OUTPUT:
            print(f"[DEBUG] rank = {torch.distributed.get_rank()} op = {self.name} output size = {list(output_tensor.size())}")
            if self.send_residual:
                print(f"[DEBUG] rank = {torch.distributed.get_rank()} op = {self.name} send residual size = {list(input_extra_tensors['input_residual'].size())}")

        return output_tensors


class ParallelBatchNormOp(RopalaModule):
    def __init__(self, 
        op_index: int, 
        in_channels: int, 
        input_width: int,        
        ## required information
        name: str = None,
        prev_name: str = None,
    ):
        super(ParallelBatchNormOp, self).__init__(op_index, name, prev_name)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.weight_size = in_channels + in_channels

        ## op-related information
        args = get_args()
        self.input_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
        self.output_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
        ## resharding information
        self.required_input_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_mats_info = {"input": {"from": "input", "trans":[0, 1, 2, 3, 4]}} 

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}
        output_tensor = self.bn(input_tensors["input"])
        output_tensors["input"] = output_tensor
        if DEBUG_OUTPUT:
            print(f"[DEBUG] rank = {torch.distributed.get_rank()} op = {self.name} output size = {list(output_tensor.size())}")

        return output_tensors


class ParallelReLUOp(RopalaModule):
    def __init__(self, 
        op_index: int, 
        in_channels: int,
        input_width: int,
        ## required information
        name: str = None,
        prev_name: str = None,
    ):
        super(ParallelReLUOp, self).__init__(op_index, name, prev_name)
        self.relu = torch.nn.ReLU(inplace=False)
        self.weight_size = 0
        ## op-related information
        args = get_args()
        self.input_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
        self.output_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
        ## resharding information 
        self.required_input_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_mats_info = {"input": {"from": "input", "trans":[0, 1, 2, 3, 4]}} 

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}
        output_tensor = self.relu(input_tensors["input"])
        output_tensors["input"] = output_tensor

        if DEBUG_OUTPUT:
            print(f"[DEBUG] rank = {torch.distributed.get_rank()} op = {self.name} output size = {list(output_tensor.size())}")

        return output_tensors

class ParallelMaxPoolOp(RopalaModule):
    def __init__(self, 
        op_index: int, 
        in_channels: int,
        input_width: int,
        kernel_size: int,
        stride: int, 
        padding: int,
        ## required information
        name: str = None,
        prev_name: str = None,
    ):
        super(ParallelMaxPoolOp, self).__init__(op_index, name, prev_name)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.weight_size = 0
        ## op-related information
        args = get_args()
        self.input_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
        self.output_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width//stride, input_width//stride], "tp_split_dim": -1, "dp_split_dim": 0}}
        ## resharding information
        self.required_input_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_mats_info = {"input": {"from": "input", "trans":[0, 1, 2, 3, 4]}} 

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}
        output_tensor = self.maxpool(input_tensors["input"])
        output_tensors["input"] = output_tensor

        if DEBUG_OUTPUT:
            print(f"[DEBUG] rank = {torch.distributed.get_rank()} op = {self.name} output size = {list(output_tensor.size())}")
        return output_tensors

class ParallelAdapAvgPoolOp(RopalaModule):
    def __init__(self, 
        op_index: int, 
        in_channels: int,
        input_width: int,
        output_width: int,
        ## required information
        name: str = None,
        prev_name: str = None,
    ):
        super(ParallelAdapAvgPoolOp, self).__init__(op_index, name, prev_name)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((output_width, output_width))
        self.weight_size = 0
        ## op-related information
        args = get_args()
        self.input_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
        self.output_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, output_width, output_width], "tp_split_dim": -1, "dp_split_dim": 0}}
        ## resharding information
        self.required_input_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_mats_info = {"input": {"from": "input", "trans":[0, 1, 2, 3, 4]}} 

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}
        output_tensor = self.avgpool(input_tensors["input"])
        output_tensors["input"] = output_tensor

        if DEBUG_OUTPUT:
            print(f"[DEBUG] rank = {torch.distributed.get_rank()} op = {self.name} output size = {list(output_tensor.size())}")

        return output_tensors        

class ParallelFCOp(RopalaModule):
    def __init__(self, 
        op_index: int, 
        in_channels: int,
        input_width: int,
        num_classes: int,
        ## required information
        name: str = None,
        prev_name: str = None,
    ):
        super(ParallelFCOp, self).__init__(op_index, name, prev_name)
        self.fc = torch.nn.Linear(in_channels, num_classes)
        self.weight_size = in_channels * num_classes + num_classes ## weight + bias
        ## op-related information
        args = get_args()
        self.input_tensors_info = {"input": {"shape": [args.micro_batch_size, in_channels, input_width, input_width], "tp_split_dim": -1, "dp_split_dim": 0}}
        self.output_tensors_info = {"input": {"shape": [args.micro_batch_size, num_classes], "tp_split_dim": -1, "dp_split_dim": 0}}
        ## resharding information
        self.required_input_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_specs = {"input": {"R": self.tp_size, "V": 1, "dims": [self.dp_size, 1, 1]} }
        self.output_mats_info = {"input": {"from": "input", "trans":[0, 1, 2, 3, 4]}} 

    def forward(self, input_tensors, input_extra_tensors, output_extra_tensors, profiling=False):
        output_tensors = {}
        input_tensor = torch.flatten(input_tensors["input"], 1)
        output_tensor = self.fc(input_tensor)
        output_tensors["input"] = output_tensor

        if DEBUG_OUTPUT:
            print(f"[DEBUG] rank = {torch.distributed.get_rank()} op = {self.name} output size = {list(output_tensor.size())}")
        return output_tensors   
