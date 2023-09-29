# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/model/module.py
# Git commit hash: 42c1cf4279acea5a554500dcb552211f44cbec45
# We retain the following copyright from the original files:

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

"""Megatron Module"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron import get_args
from megatron import mpu


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)



def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared



class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings

        ## flexpipe
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}
        self.shared_weights_info = {}        
        self.is_last_op = False
        ## resharding
        self.output_extra_specs = None
        self.output_extra_mats_info = None
        self.required_input_extra_specs = {}
        self.input_extra_mats = None
        self.new_input_extra_tensors = {}
        self.tmp_buffer = None
        self.elementwise = False
        self.input_mats = None
        self.input_extra_mats = None

    def parse_op_configs(self, config):
        self.name = config.name
        self.prev_name = config.prev_name
        self.input_tensors_info = config.input_tensors_info
        self.output_tensors_info = config.output_tensors_info
        self.input_extra_tensors_info = config.input_extra_tensors_info
        self.output_extra_tensors_info = config.output_extra_tensors_info        
        self.shared_weights_info = config.shared_weights_info
        
    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)


    def word_embeddings_weight(self):
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            # return self.language_model.embedding.word_embeddings.weight
            return self.language_model.ops[0].word_embeddings.weight
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if not self.share_word_embeddings:
                raise Exception('word_embeddings_weight() called for last '
                                'stage, but share_word_embeddings is false')

            return self.word_embeddings.weight
        raise Exception('word_embeddings_weight() should be '
                        'called for first and last stage only')


    # def initialize_word_embeddings(self, init_method_normal):
    #     args = get_args()
    #     if not self.share_word_embeddings:
    #         raise Exception('initialize_word_embeddings() was called but '
    #                         'share_word_embeddings is false')

    #     # This function just initializes the word embeddings in the final stage
    #     # when we are using pipeline parallelism. If we aren't using pipeline
    #     # parallelism there is nothing to do.
    #     if args.pipeline_model_parallel_size == 1:
    #         return

    #     # Parameters are shared between the word embeddings layer, and the
    #     # heads at the end of the model. In a pipelined setup with more than
    #     # one stage, the initial embedding layer and the head are on different
    #     # workers, so we do the following:
    #     # 1. Create a second copy of word_embeddings on the last stage, with
    #     #    initial parameters of 0.0.
    #     # 2. Do an all-reduce between the first and last stage to ensure that
    #     #    the two copies of word_embeddings start off with the same
    #     #    parameter values.
    #     # 3. In the training loop, before an all-reduce between the grads of
    #     #    the two word_embeddings layers to ensure that every applied weight
    #     #    update is the same on both stages.
    #     if mpu.is_pipeline_last_stage():
    #         assert not mpu.is_pipeline_first_stage()
    #         self._word_embeddings_for_head_key = 'word_embeddings_for_head'
    #         # set word_embeddings weights to 0 here, then copy first
    #         # stage's weights using all_reduce below.
    #         self.word_embeddings = mpu.VocabParallelEmbedding(
    #             args.padded_vocab_size, args.hidden_size,
    #             init_method=init_method_normal(args.init_method_std))
    #         self.word_embeddings.weight.data.fill_(0)
    #         self.word_embeddings.weight.shared = True

    #     # Ensure that first and last stages have the same initial parameter
    #     # values.
    #     if torch.distributed.is_initialized():
    #         if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
    #             torch.distributed.all_reduce(self.word_embeddings_weight().data,
    #                                          group=mpu.get_embedding_group())
    #     else:
    #         print("WARNING! Distributed processes aren't initialized, so "
    #               "word embeddings in the last layer are not initialized. "
    #               "If you are just manipulating a model this is fine, but "
    #               "this needs to be handled manually. If you are training "
    #               "something is definitely wrong.")

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_word_embeddings is false')
        # Parameters are shared between the word embeddings layer, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if mpu.is_pipeline_last_stage():
            if not mpu.is_pipeline_first_stage():
                self._word_embeddings_for_head_key = 'word_embeddings_for_head'
                # If first and last stages are different, set word_embeddings
                # weights to 0 here, then copy first stage's weights using
                # all_reduce below.
                self.word_embeddings = mpu.VocabParallelEmbedding(
                    args.padded_vocab_size, args.hidden_size,
                    init_method=init_method_normal(args.init_method_std))
                self.word_embeddings.weight.data.fill_(0)
                self.word_embeddings.weight.shared = True
        # Ensure that first and last stages have the same initial parameter values.
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            torch.distributed.all_reduce(self.word_embeddings_weight().data,
                                        group=mpu.get_embedding_group())

class RopalaModule(torch.nn.Module):
    def __init__(self, op_index, name, prev_name, is_last_op=False):
        super(RopalaModule, self).__init__()
        self.name = name
        self.prev_name = prev_name
        self.op_index = op_index
        self.is_last_op = is_last_op

        self.tp_size = mpu.get_op_tp_size(op_index)
        self.dp_size = mpu.get_op_dp_size(op_index)

        self.input_tensors_info = {}
        self.output_tensors_info = {}
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}
        self.shared_weights_info = {}        
        
        ## resharding
        self.output_extra_specs = None
        self.output_extra_mats_info = None
        self.required_input_extra_specs = {}
        self.input_extra_mats = None
        self.new_input_extra_tensors = {}
        self.tmp_buffer = None
        self.elementwise = False
        self.input_mats = None
        self.input_extra_mats = None

        ## for profiling
        self.weight_size = 0

    def get_shared_tensor(self, grads=False):
        args = get_args()
        tensor_dict = {}
        for key in sorted(self.shared_weights_info):
            if key == "word_embeddings":
                if grads:
                    if args.DDP_impl == 'local':
                        tensor_dict["word_embeddings"] = self.word_embeddings.weight.main_grad
                    else:
                        tensor_dict["word_embeddings"] = self.word_embeddings.weight.grad
                else:
                    tensor_dict["word_embeddings"] = self.word_embeddings.weight.data
            elif key == "position_embeddings":
                if grads:
                    if args.DDP_impl == 'local':
                        tensor_dict["position_embeddings"] = self.position_embeddings.weight.main_grad
                    else:
                        tensor_dict["position_embeddings"] = self.position_embeddings.weight.grad
                else:
                    tensor_dict["position_embeddings"] = self.position_embeddings.weight.data          
        return tensor_dict

    def set_shared_tensor(self, new_data, grads=False):
        args = get_args()
        for key in sorted(self.shared_weights_info):
            if key == "word_embeddings":
                if grads:
                    if args.DDP_impl == 'local':
                        self.word_embeddings.weight.main_grad = new_data["word_embeddings"][0]
                    else:
                        self.word_embeddings.weight.grad = new_data["word_embeddings"][0]
                else:
                    self.word_embeddings.weight.data = new_data["word_embeddings"][0]
            elif key == "position_embeddings":
                if grads:
                    if args.DDP_impl == 'local':
                        self.position_embeddings.weight.main_grad = new_data["position_embeddings"][0]
                    else:
                        self.position_embeddings.grad = new_data["position_embeddings"][0]
                else:
                    self.position_embeddings.weight.data = new_data["position_embeddings"][0]

def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val
    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)



class Float16Module(MegatronModule):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()

        if args.fp16:
            self.add_module('module', module.half())
            def float16_convertor(val):
                return val.half()
        elif args.bf16:
            self.add_module('module', module.bfloat16())
            def float16_convertor(val):
                return val.bfloat16()
        else:
            raise Exception('should not be here')

        self.float16_convertor = float16_convertor


    def forward(self, *inputs, **kwargs):
        if mpu.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if mpu.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
