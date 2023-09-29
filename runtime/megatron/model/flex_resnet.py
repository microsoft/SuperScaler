# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Type, Any, Callable, Union, List, Optional, Dict, TypeVar

import torch
import torch.nn as nn
import numpy as np
from .module import MegatronModule
from megatron import get_args, mpu
from .flex_model import get_flex_model
from .flex_ops import gen_op
import os 

op_start_index = 0 
op_end_index = 0

def _resnet_block(current_op_index, current_width, in_channels, base_channels, width_factor, stride=1, layer_index=0, block_index=0, downsample=False, prev_name=None):
    expansion = 4

    residual_in_channels = in_channels
    residual_input_width = current_width

    tmp_op_list = []
    block_name = "block-" + str(layer_index) + "-" + str(block_index)
    tmp_op_list.append({"name": block_name+"-conv1", "op_index": current_op_index, "input_width": current_width, "in_channels": in_channels, "out_channels": base_channels, "kernel_size": 1, "stride": 1, "padding": 0, "send_residual": True, "prev_name": prev_name})
    tmp_op_list.append({"name": block_name+"-bn1", "op_index": current_op_index+1, "input_width": current_width, "in_channels": base_channels, "prev_name": block_name+"-conv1"})
    tmp_op_list.append({"name": block_name+"-conv2", "op_index": current_op_index+2, "input_width": current_width, "in_channels": base_channels, "out_channels": base_channels * width_factor, "kernel_size": 3, "stride": stride, "padding": 1, "send_residual": False, "prev_name": block_name+"-bn1"})
    tmp_op_list.append({"name": block_name+"-bn2", "op_index": current_op_index+3, "input_width": current_width//stride, "in_channels": base_channels * width_factor, "prev_name": block_name+"-conv2"})
    tmp_op_list.append({"name": block_name+"-conv3", "op_index": current_op_index+4, "input_width": current_width//stride, "in_channels": base_channels * width_factor, "out_channels": base_channels * expansion, "kernel_size": 1, "stride": 1, "padding": 0, "send_residual": False, "prev_name": block_name+"-bn2"})
    tmp_op_list.append({"name": block_name+"-bn3", "op_index": current_op_index+5, "input_width": current_width//stride, "in_channels": base_channels * expansion, "prev_name": block_name+"-conv3"})

    if downsample:
        down_sample_kernel_size = 1
    else:
        down_sample_kernel_size = 0

    tmp_op_list.append({"name": block_name+"-downsample", "op_index": current_op_index+6, "input_width": current_width//stride, "in_channels": base_channels * expansion, "out_channels": base_channels * expansion, "kernel_size": down_sample_kernel_size, "stride": stride, "padding": 0, "downsample": downsample, "recv_residual": True, "residual_in_channels": residual_in_channels, "residual_input_width": residual_input_width, "prev_name": block_name+"-bn3"})
    tmp_op_list.append({"name": block_name+"-relu", "op_index": current_op_index+7, "input_width": current_width//stride, "in_channels": base_channels * expansion, "prev_name": block_name+"-downsample"})

    next_op_index = current_op_index+8
    return tmp_op_list, next_op_index, current_width//stride, base_channels * expansion, block_name+"-relu"


def _resnet(num_layers_list, initial_width, base_channels, width_factor):
    expansion = 4

    op_list = []
    op_list.append({"name": "conv1", "op_index": 0, "input_width": initial_width, "in_channels": 3, "out_channels": base_channels, "kernel_size": 7, "stride": 2, "padding": 3, "send_residual": False,  "prev_name": None})
    op_list.append({"name": "bn1", "op_index": 1, "input_width": initial_width//2, "in_channels": base_channels, "prev_name": "conv1"})
    op_list.append({"name": "relu1", "op_index": 2, "input_width": initial_width//2, "in_channels": base_channels, "prev_name": "bn1"})
    op_list.append({"name": "maxpool", "op_index": 3, "input_width": initial_width//2, "in_channels": base_channels, "kernel_size": 3, "stride": 2, "padding": 1, "prev_name": "relu1"})

    current_width = initial_width//4
    current_op_index = 4
    current_in_channels = base_channels
    prev_name = "maxpool"
    for i in range(len(num_layers_list)):
        _base_channels = base_channels * 2**i
        num_blocks = num_layers_list[i]
        if i == 0:
            stride = 1
        else:
            stride = 2
        tmp_op_list, current_op_index, current_width, current_in_channels, prev_name = _resnet_block(current_op_index, current_width, current_in_channels, _base_channels, width_factor, layer_index=i, block_index=0, stride=stride, downsample=True, prev_name=prev_name)
        op_list += tmp_op_list
        for block_index in range(1, num_blocks):
            tmp_op_list, current_op_index, current_width, current_in_channels, prev_name = _resnet_block(current_op_index, current_width, current_in_channels, _base_channels, width_factor, layer_index=i, block_index=block_index, prev_name=prev_name)
            op_list += tmp_op_list
    
    op_list.append({"name": "avgpool", "op_index": current_op_index, "input_width": current_width, "in_channels": current_in_channels, "output_width": 1, "prev_name": prev_name})
    op_list.append({"name": "fc", "op_index": current_op_index+1, "input_width": 1, "in_channels": current_in_channels, "num_classes": 1024, "prev_name": "avgpool"})
    return op_list

class FlexResNet(MegatronModule):
    def __init__(self, num_layers_list=None, in_channels=None, width_factor=None, pre_process=True, post_process=True, profiling=False):
        super(FlexResNet, self).__init__()
        initial_width = 224

        if not profiling:
            args = get_args()
            num_layers_list = args.num_layers
            in_channels = args.in_channels
            width_factor = args.width_factor

            global op_start_index, op_end_index
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
            virtual_pipeline_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            op_start_index = mpu.get_op_start_index(pipeline_rank, virtual_pipeline_rank)
            op_end_index = mpu.get_op_end_index(pipeline_rank, virtual_pipeline_rank)
            num_ops = op_end_index - op_start_index
            assert num_ops >= 0

            full_op_list = _resnet(num_layers_list, initial_width, in_channels, width_factor)
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
            self.full_op_list = _resnet(num_layers_list, initial_width, in_channels, width_factor)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_tensors, input_extra_tensors):    
        output_tensor = self.language_model(input_tensors, input_extra_tensors)
        return output_tensor
