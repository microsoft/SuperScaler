# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/model/__init__.py
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

from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm

from .distributed import DistributedDataParallel
from .module import Float16Module
from .flex_gpt import FlexGPTModel
from .flex_t5 import FlexT5Model
from .flex_resnet import FlexResNet