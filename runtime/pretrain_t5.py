# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/pretrain_t5.py
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

"""Pretrain T5"""

from functools import partial

import torch

from megatron import (
    get_args,
    get_timers,
    mpu,
    print_rank_0
)
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import FlexT5Model
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building T5 model ...')

    model = FlexT5Model(num_tokentypes=0,
                    parallel_output=True,
                    pre_process=pre_process,
                    post_process=post_process)

    return model

def t5_extended_attention_mask(attention_mask_list):

    def attn_mask_postprocess(attn_mask):
        # [b, 1, s, s]
        extended_attention_mask = attn_mask.unsqueeze(1)
        return extended_attention_mask

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]


def t5_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids

def get_batch(data_iterator):
    """Build the batch."""
    
    ## Currently using synthetic data
    args = get_args()
    vocab_size = 30624
    tokens_enc = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(0), args.encoder_seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    # tokens_dec = torch.rand((args.micro_batch_size, args.decoder_seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((args.micro_batch_size//mpu.get_op_dp_size(-1), args.decoder_seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    labels = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(-1), args.decoder_seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    enc_mask = (torch.rand((args.micro_batch_size, args.encoder_seq_length, args.encoder_seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    dec_mask = (torch.rand((args.micro_batch_size, args.decoder_seq_length, args.decoder_seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    enc_dec_mask = (torch.rand((args.micro_batch_size, args.decoder_seq_length, args.encoder_seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5)

    return tokens_enc, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask


def loss_func(loss_mask, output_tensor):
    
    lm_loss_ = output_tensor["output"].float()

    lm_loss_ = lm_loss_.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])

    return loss, {'lm loss': averaged_losses[0]}

def forward_step(data_iterator, model, extra_tensors_):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens_enc, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask = get_batch(data_iterator)

    # Converting the attention masks to proper parameter settings
    encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = t5_extended_attention_mask([enc_mask, dec_mask, enc_dec_mask])
    encoder_position_ids = t5_position_ids(tokens_enc)
    # decoder_position_ids = t5_position_ids(tokens_dec)

    input_tensors = {}
    input_tensors["enc_input_ids"] = tokens_enc
    input_tensors["enc_position_ids"] = encoder_position_ids
    # input_tensors["dec_input_ids"] = tokens_dec
    # input_tensors["dec_position_ids"] = decoder_position_ids

    extra_tensors = {}
    extra_tensors["labels"] = lm_labels
    extra_tensors["enc_attention_mask"] = encoder_attn_mask
    extra_tensors["dec_attention_mask"] = decoder_attn_mask
    extra_tensors["enc_dec_attention_mask"] = encoder_decoder_attn_mask

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
    #              'for T5 ...')
    # train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    #     data_prefix=args.data_path,
    #     data_impl=args.data_impl,
    #     splits_string=args.split,
    #     train_valid_test_num_samples=train_val_test_num_samples,
    #     max_seq_length=args.encoder_seq_length,
    #     max_seq_length_dec=args.decoder_seq_length,
    #     masked_lm_prob=args.mask_prob,
    #     short_seq_prob=args.short_seq_prob,
    #     seed=args.seed,
    #     skip_warmup=(not args.mmap_warmup),
    #     dataset_type='t5')
    # print_rank_0("> finished creating T5 datasets ...")
    # return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    forward_step_func = forward_step
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step_func,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
