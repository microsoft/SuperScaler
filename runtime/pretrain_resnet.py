# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/pretrain_gpt.py
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

"""Pretrain wide-resnet"""

import torch

from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import FlexResNet
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
import torch.nn.functional as F


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building ResNet model ...')

    model = FlexResNet(pre_process=pre_process, post_process=post_process)
    return model


def get_batch(data_iterator):
    """Generate a batch"""

    ## Currently using synthetic data
    args = get_args()
    if args.fp16:
        input_images = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(0), 3, 224, 224), requires_grad=True, device=torch.cuda.current_device(), dtype=torch.float16)
    else:
        input_images = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(0), 3, 224, 224), requires_grad=True, device=torch.cuda.current_device(), dtype=torch.float32)
    labels = torch.randint(1024, (args.micro_batch_size//mpu.get_op_dp_size(-1),), device=torch.cuda.current_device(), dtype=torch.int64)

    return input_images, labels

def loss_func(labels, output_tensor):

    losses = output_tensor["input"].float()
    loss = F.cross_entropy(losses, labels)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model, extra_tensors_):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    input_tensor, labels = get_batch(data_iterator)
    input_tensors = {"input": input_tensor}
    extra_tensors = {}
    if extra_tensors_ is not None:
        for key in extra_tensors_:
            extra_tensors[key] = extra_tensors_[key]

    timers('batch-generator').stop()

    if mpu.is_pipeline_last_stage():
        output_tensor = model(input_tensors, extra_tensors)
        ouput_extra_tensors = None
    else:
        output_tensor, ouput_extra_tensors = model(input_tensors, extra_tensors)

    return output_tensor, ouput_extra_tensors, partial(loss_func, labels)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    ## Currently using synthetic data
    return None, None, None


if __name__ == "__main__":
    forward_step_func = forward_step
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step_func,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
