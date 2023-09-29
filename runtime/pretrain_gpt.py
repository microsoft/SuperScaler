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

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import FlexGPTModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = FlexGPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""

    ## Currently using synthetic data
    args = get_args()
    vocab_size = 50257
    tokens = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(0), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    # labels = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(-1), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((args.micro_batch_size//mpu.get_op_dp_size(-1), args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    attention_mask = (torch.rand((args.micro_batch_size, 1, args.seq_length, args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    position_ids = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(0), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * args.seq_length

    return tokens, loss_mask, position_ids, attention_mask

def loss_func(loss_mask, output_tensor):
    losses = output_tensor["output"].float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}

def forward_step(data_iterator, model, extra_tensors_):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch. 
    timers('batch-generator').start()
    tokens, loss_mask, position_ids, attention_mask = get_batch(data_iterator)
    input_tensors = {}
    input_tensors["enc_input_ids"] = tokens
    input_tensors["enc_position_ids"] = position_ids
    extra_tensors = {}
    extra_tensors["enc_attention_mask"] = attention_mask
    # extra_tensors["labels"] = labels
    if extra_tensors_ is not None:
        for key in extra_tensors_:
            extra_tensors[key] = extra_tensors_[key]
            
    timers('batch-generator').stop()

    if mpu.is_pipeline_last_stage():
        output_tensor = model(input_tensors, extra_tensors)
        ouput_extra_tensors = None
    else:
        output_tensor, ouput_extra_tensors = model(input_tensors, extra_tensors)

    return output_tensor, ouput_extra_tensors, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    ## Currently using synthetic data
    return None, None, None

    # args = get_args()
    # print_rank_0('> building train, validation, and test datasets '
    #             'for GPT ...')
    # train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    #     data_prefix=args.data_path,
    #     data_impl=args.data_impl,
    #     splits_string=args.split,
    #     train_valid_test_num_samples=train_val_test_num_samples,
    #     seq_length=args.seq_length,
    #     seed=args.seed,
    #     skip_warmup=(not args.mmap_warmup))
    # print_rank_0("> finished creating GPT datasets ...")

    # return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    forward_step_func = forward_step

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step_func,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
