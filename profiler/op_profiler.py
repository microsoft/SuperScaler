# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import time
import csv
import pickle
import os 
import numpy as np 
import sys
sys.path.append("../runtime/")
from megatron.initialize import initialize_megatron
from megatron import get_args, mpu
from megatron.model.flex_resnet import FlexResNet
from megatron.model.flex_gpt import FlexGPTModel
from megatron.model.flex_t5 import FlexT5Model
from megatron.model.flex_ops import gen_op
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import unwrap_model
from megatron.model import Float16Module
from megatron.utils import debug_mem_report
from megatron.utils import report_memory
from model_configs import model_prof_configs, resnet_configs, gpt_configs, t5_configs

DATA_BASE = 4/(1024 * 1024)
SKIP_RUNNING = os.environ.get("SKIP_RUNNING", '0') == '1'

## shapes for input tensors and extra tensors
input_shape_dict = {}
input_extra_dict = {}
## size of inputs and outputs
input_size_dict = {}
output_size_dict = {}
activation_size_dict = {}
weight_size_dict = {}

def print_rank0(str):
    if torch.distributed.get_rank() == 0:
        print(str)

def print_cached_dicts(cached_dict):
    for item in cached_dict:
        print(f"{item}: {cached_dict[item]}")

def wrap_op(op):
    args = get_args()
    for key in op.output_extra_tensors_info:
        op.output_extra_tensors_info[key]["cross_stage"] = True
    all_ranks = mpu.get_ranks_via_pipeline_stage(mpu.get_pipeline_model_parallel_rank())
    input_mats = np.array(all_ranks).reshape([1, 1, 1, op.dp_size, op.tp_size])    
    input_mats_ = {}
    for name in op.required_input_specs:
        input_mats_[name] = input_mats
    op.input_mats = input_mats_

    for param in op.parameters():
        mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    op.cuda(torch.cuda.current_device())
    if args.fp16:
        op = Float16Module(op, args) 

    if args.DDP_impl == 'local':
        op = LocalDDP(op,
                          args.accumulate_allreduce_grads_in_fp32,
                          args.use_contiguous_buffers_in_ddp)
        return op
    # return op
    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))

def get_params_dtype(params_dtype):
    global DATA_BASE
    args = get_args()

    if params_dtype == "fp32":
        DATA_BASE = 4 / (1024*1024)
        params_dtype = torch.float
        args.fp16 = False
        args.params_dtype = params_dtype
    elif params_dtype == "fp16":
        DATA_BASE = 2 / (1024*1024)
        params_dtype = torch.half
        args.fp16 = True
        args.params_dtype = params_dtype
    else:
        raise RuntimeError(f"data type {params_dtype} not supported.")
    return params_dtype

def get_model(model_name, model_size):
    
    args = get_args()
    if model_name == "resnet":
        num_layers_list, base_channels, width_factor, params_dtype = resnet_configs[model_size]
        params_dtype = get_params_dtype(params_dtype)
        model = FlexResNet(num_layers_list=num_layers_list, in_channels=base_channels, width_factor=width_factor, profiling=True)
    elif model_name == "gpt":
        num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype = gpt_configs[model_size]
        params_dtype = get_params_dtype(params_dtype)
        args.seq_length = seq_len
        args.hidden_size = hidden_size
        args.ffn_hidden_size = ffn_hidden_size
        args.num_attention_heads = num_attention_heads
        args.kv_channels = kv_channels
        args.max_position_embeddings = seq_len
        args.padded_vocab_size = vocab_size
        args.num_layers = num_layers
        args.seq_length = seq_len
        model = FlexGPTModel(num_layers=num_layers, hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size, num_attention_heads=num_attention_heads, kv_channels=kv_channels, profiling=True)
    elif model_name == "t5":
        num_layers, encoder_seq_length, decoder_seq_length, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype = t5_configs[model_size]
        params_dtype = get_params_dtype(params_dtype)
        args.encoder_seq_length = encoder_seq_length
        args.decoder_seq_length = decoder_seq_length
        args.seq_length = encoder_seq_length
        args.hidden_size = hidden_size
        args.ffn_hidden_size = ffn_hidden_size
        args.num_attention_heads = num_attention_heads
        args.kv_channels = kv_channels
        args.max_position_embeddings = encoder_seq_length
        args.padded_vocab_size = vocab_size
        args.num_layers = num_layers
        args.resharding_stages = [False]
        model = FlexT5Model(profiling=True)

    args.model_name = model_name
    return model

def infer_data_size(op_list, save_filename_prefix, mbs, algo):
    '''
    Infer each op's input/output tensor shape, which will be used to generate input/output tensor during the profiling.
    '''
    global input_size_dict, output_size_dict, weight_size_dict, activation_size_dict, input_shape_dict, input_extra_dict
    args = get_args()
    tp_size = args.prof_tp_size

    prev_extra_size = 0
    for op_info in op_list:
        op_info["op_index"] = 0
        op_uniq_name = save_filename_prefix + op_info["name"] + f"mbs{mbs}tp_size{tp_size}algo{algo}"
        op = unwrap_model(gen_op(op_info, algo=algo), (LocalDDP, Float16Module)) 

        weight_size_dict[op_uniq_name] = np.prod(op.weight_size) * DATA_BASE
        sum_input_size = 0
        sum_output_size = 0
        _input_shape_dict = {}
        _input_extra_dict = {}

        ## infer input tensor size
        for input_name in op.input_tensors_info:
            input_shape = op.input_tensors_info[input_name]["shape"]
            tp_split_dim = op.input_tensors_info[input_name]["tp_split_dim"]
            if len(op.required_input_specs[input_name]) == 0 or op.required_input_specs[input_name]["R"] == op.tp_size:
                apply_tp = False
            elif op.required_input_specs[input_name]["R"] == 1:
                apply_tp = True
            else:
                raise RuntimeError(f"not supported input spec for op ({input_name}): {op.required_input_specs[input_name]}")
            if apply_tp:
                input_shape[tp_split_dim] //= op.tp_size
            
            _input_shape_dict[input_name] = input_shape
            sum_input_size += np.prod(input_shape) * DATA_BASE
        sum_input_size += prev_extra_size

        ## infer output tensor size
        for output_name in op.output_tensors_info:
            output_shape = op.output_tensors_info[output_name]["shape"]
            tp_split_dim = op.output_tensors_info[output_name]["tp_split_dim"]
            if len(op.output_specs[output_name]) == 0 or op.output_specs[output_name]["R"] == op.tp_size:
                apply_tp = False
            elif op.output_specs[output_name]["R"] == 1:
                apply_tp = True
            else:
                raise RuntimeError(f"not supportted input spec for op ({output_name}): {op.output_specs[output_name]}")
            if apply_tp:
                output_shape[tp_split_dim] //= op.tp_size
            sum_output_size += np.prod(output_shape) * DATA_BASE
        ## save the inferred size
        activation_size_dict[op_uniq_name] = sum_output_size

        ## infer input extra tensor size
        sum_input_extra_size = 0
        for input_extra_name in op.input_extra_tensors_info:
            input_shape = op.input_extra_tensors_info[input_extra_name]["shape"]
            tp_split_dim = op.input_extra_tensors_info[input_extra_name]["tp_split_dim"]
            if op.required_input_extra_specs[input_extra_name]["R"] == op.tp_size:
                apply_tp = False
            elif op.required_input_extra_specs[input_extra_name]["R"] == 1:
                apply_tp = True
            else:
                raise RuntimeError(f"not supportted input spec for op ({input_extra_name}): {op.required_input_extra_specs[input_extra_name]}")
            if apply_tp:
                input_shape[tp_split_dim] //= op.tp_size
            _input_extra_dict[input_extra_name] = input_shape
            ## current workaround for masks.
            if "mask" not in input_extra_name:
                sum_input_extra_size += np.prod(input_shape) * DATA_BASE

        ## infer output extra tensor size
        sum_output_extra_size = 0
        for output_extra_name in op.output_extra_tensors_info:
            output_shape = op.output_extra_tensors_info[output_extra_name]["shape"]
            tp_split_dim = op.output_extra_tensors_info[output_extra_name]["tp_split_dim"]
            if op.output_extra_specs[output_extra_name]["R"] == op.tp_size:
                apply_tp = False
            elif op.output_extra_specs[output_extra_name]["R"] == 1:
                apply_tp = True
            else:
                raise RuntimeError(f"not supportted output spec for op ({output_extra_name}): {op.output_extra_specs[output_extra_name]}")
            if apply_tp:
                output_shape[tp_split_dim] //= op.tp_size
            sum_output_extra_size += np.prod(output_shape) * DATA_BASE

        current_extra_size = prev_extra_size + sum_output_extra_size - sum_input_extra_size
        print(f"{op_info['name']}: current_extra_size = {current_extra_size}= prev_extra_size {prev_extra_size}+ sum_output_extra_size {sum_output_extra_size}- sum_input_extra_size {sum_input_extra_size}")
        sum_output_size += current_extra_size
        prev_extra_size = current_extra_size

        ## save the inferred size
        input_size_dict[op_uniq_name] = sum_input_size
        output_size_dict[op_uniq_name] = sum_output_size
        input_shape_dict[op_uniq_name] = _input_shape_dict
        input_extra_dict[op_uniq_name] = _input_extra_dict

def get_inputs(op_uniq_name, params_dtype):
    inputs = {}
    input_extra_tensors = {}
    for input_name in input_shape_dict[op_uniq_name]:
        input_shape = input_shape_dict[op_uniq_name][input_name]
        if input_name in ["enc_input_ids", "enc_position_ids"]:
            inputs[input_name] = torch.rand(input_shape, requires_grad=False, device=torch.cuda.current_device()).long() 
        else:
            inputs[input_name] = torch.rand(input_shape, requires_grad=True, device=torch.cuda.current_device(), dtype=params_dtype)        

    for input_extra_name in input_extra_dict[op_uniq_name]:
        input_shape = input_extra_dict[op_uniq_name][input_extra_name]
        if input_extra_name in ["enc_attention_mask", "dec_attention_mask", "enc_dec_attention_mask"]:
            input_extra_tensors[input_extra_name] = (torch.rand(input_shape, requires_grad=False, device=torch.cuda.current_device(), dtype=params_dtype) < 0.5)
        else:
            input_extra_tensors[input_extra_name] = torch.rand(input_shape, requires_grad=True, device=torch.cuda.current_device(), dtype=params_dtype)

    return inputs, input_extra_tensors

## this function is used for resnet, to find same operators.
## for GPT and T5, no need to hash op, because the possiblities are less.
def get_op_hash(op_info, micro_batch_size, tp_size, algo, save_filename_prefix):
    hash_str = save_filename_prefix + "mbs" + str(micro_batch_size) + "tp" + str(tp_size)
    args = get_args()
    
    if args.model_name == "resnet":
        for op_type in ["conv", "downsample", "bn", "relu", "maxpool", "avgpool", "fc"]:
            if op_type in op_info["name"]:
                current_op_type = op_type
                hash_str += op_type
        for item in op_info:
            if item not in ["name", "prev_name", "op_index"]:
                hash_str +=  str(op_info[item])
        if current_op_type in ["conv", "downsample"]:
            hash_str += "_algo" + str(algo)        
    else:
        hash_str += op_info["name"]
        for gemm_op_name in ["qkv", "dense", "GEMM"]:
            if gemm_op_name in op_info["name"]:
                hash_str += "_algo" + str(algo)

    return hash_str

def get_input_tensors(op_info, input_data, input_extra_tensors):
    if op_info["name"] == "encoder-embedding":
        input_tensors = None
    else:
        input_tensors = []
        for input_name in input_data:
            input_tensors.append(input_data[input_name])
        for input_extra_name in input_extra_tensors:
            ## workaround for softmax op.
            if "mask" not in input_extra_name and "bias" not in input_extra_name: 
                input_tensors.append(input_extra_tensors[input_extra_name])   

    return input_tensors 

def get_outputs_and_grads(output_tensors, output_extra_tensors, grad_type):
    ## keep original output tensors 
    origin_outputs = []
    for output_name in output_tensors:
        origin_outputs.append(output_tensors[output_name])
    for output_extra_name in output_extra_tensors:
        origin_outputs.append(output_extra_tensors[output_extra_name])

    output_grads = []
    ## add one more dummy op for each output tensor
    for output_tensor in origin_outputs:
        tensor_shape = list(output_tensor.size())
        if len(tensor_shape) >= 3:
            pool_op = torch.nn.AdaptiveMaxPool2d(1)
        elif len(tensor_shape) >= 2:
            pool_op = torch.nn.AdaptiveMaxPool1d(1)
        else:
            pool_op = torch.nn.Identity()
        output_tensor_ = pool_op(output_tensor)
        output_tensor_grad = torch.randn(output_tensor_.size(), requires_grad=False, device=torch.cuda.current_device(), dtype=grad_type)
        origin_grad = torch.autograd.grad(outputs=output_tensor_, grad_outputs=output_tensor_grad, inputs=output_tensor, allow_unused=False, retain_graph=False)
        output_grads.append(origin_grad[0])    

    return origin_outputs, output_grads

def profile_op(mbs, algo, op_info, params_dtype, grad_type, op_uniq_name):
    global profiled_results
    op_info["op_index"] = 0
    op = wrap_op(gen_op(op_info, algo=algo))
    input_data, input_extra_tensors = get_inputs(op_uniq_name, params_dtype)
    input_tensors = get_input_tensors(op_info, input_data, input_extra_tensors)
    output_extra_tensors = {}

    ## Profiling forward/backward computation time
    sum_fwd_time = 0
    sum_bwd_time = 0
    if op_info["name"] in ["gpt-post-process", "t5-post-process"]:
        for index in range(args.prof_repeat_times[0] + args.prof_warmup_times):
            torch.cuda.synchronize()
            start_time = time.time()
            output_data = op(input_data, input_extra_tensors, output_extra_tensors, profiling=True)
            torch.cuda.synchronize()
            end_time = time.time()   
            if index >= args.prof_warmup_times:
                sum_fwd_time += end_time - start_time     
            outputs, output_grads = get_outputs_and_grads(output_data, output_extra_tensors, grad_type)

            torch.cuda.synchronize()
            start_time = time.time()
            torch.autograd.grad(outputs=outputs, grad_outputs=output_grads, inputs=input_tensors, allow_unused=False, retain_graph=True)
            torch.cuda.synchronize()
            end_time = time.time()   
            if index >= args.prof_warmup_times:
                sum_bwd_time += end_time - start_time
        avg_fwd_time = sum_fwd_time * 1000000 / args.prof_repeat_times[0]
        avg_bwd_time = sum_bwd_time * 1000000 / args.prof_repeat_times[0]
    else:
        ## warm-up
        torch.cuda.synchronize()
        start_time = time.time()
        for index in range(args.prof_warmup_times):
            output_data = op(input_data, input_extra_tensors, output_extra_tensors, profiling=True)
        torch.cuda.synchronize()
        end_time = time.time()   
        sum_warmup_time = end_time - start_time    

        avg_warmup_time = (sum_warmup_time * 1000000) / args.prof_warmup_times
        tensor_value = torch.tensor(avg_warmup_time).cuda()
        torch.distributed.all_reduce(tensor_value, op=torch.distributed.ReduceOp.AVG)
        avg_warmup_time = tensor_value.item()

        if args.prof_repeat_threshold is not None and avg_warmup_time >= args.prof_repeat_threshold:
            remaining_fwd_times = args.prof_repeat_times[1]
            remaining_bwd_times = args.prof_repeat_times[1]    
        else:
            remaining_fwd_times = args.prof_repeat_times[0]
            remaining_bwd_times = args.prof_repeat_times[0]

        if args.prof_warmup_threshold is not None and avg_warmup_time >= args.prof_warmup_threshold:
            remaining_fwd_times = max(remaining_fwd_times - args.prof_warmup_times, 0)
            sum_fwd_time += sum_warmup_time

        ##### forward, sync after all runs
        torch.cuda.synchronize()
        start_time = time.time()
        for index in range(remaining_fwd_times):
            output_data = op(input_data, input_extra_tensors, output_extra_tensors, profiling=True)          
        torch.cuda.synchronize()
        end_time = time.time()   
        sum_fwd_time += end_time - start_time            
        if args.prof_warmup_threshold is not None and avg_warmup_time >= args.prof_warmup_threshold:
            avg_fwd_time = sum_fwd_time * 1000000 / (remaining_fwd_times + args.prof_warmup_times)
        else:
            avg_fwd_time = sum_fwd_time * 1000000 / remaining_fwd_times

        origin_outputs, output_grads = get_outputs_and_grads(output_data, output_extra_tensors, grad_type) 

        ### one-time warmup for backward
        if op_info["name"] == "encoder-embedding":
            torch.autograd.backward(origin_outputs, grad_tensors=output_grads, retain_graph=True)
        else:
            torch.autograd.grad(outputs=origin_outputs, grad_outputs=output_grads, inputs=input_tensors, allow_unused=False, retain_graph=True)

        ## backward, sync after all run
        torch.cuda.synchronize()
        start_time = time.time()
        for index in range(remaining_bwd_times):
            if op_info["name"] == "encoder-embedding":
                torch.autograd.backward(origin_outputs, grad_tensors=output_grads, retain_graph=True)
            else:
                torch.autograd.grad(outputs=origin_outputs, grad_outputs=output_grads, inputs=input_tensors, allow_unused=False, retain_graph=True)
        torch.cuda.synchronize()
        end_time = time.time()   
        sum_bwd_time += end_time - start_time
        avg_bwd_time = sum_bwd_time * 1000000 / remaining_bwd_times

    ## Profiling memory
    _mem_reserved_fwd = 0 
    _mem_reserved_bwd = 0 
    _mem_allocated = 0

    output_data = None
    input_data = None
    input_tensors = None
    origin_outputs = None
    output_grads = None
    input_extra_tensors = None
    output_extra_tensors = None
    torch.cuda.empty_cache()

    input_data, input_extra_tensors = get_inputs(op_uniq_name, params_dtype)
    output_extra_tensors = {}

    mem_allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    mem_reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)

    output_data = op(input_data, input_extra_tensors, output_extra_tensors, profiling=True)

    new_mem_allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    new_mem_reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)

    _mem_reserved_fwd = new_mem_reserved - mem_reserved
    _mem_allocated = new_mem_allocated - mem_allocated
    _mem_reserved_fwd -= _mem_allocated
    if _mem_reserved_fwd < 0:
        _mem_reserved_fwd = 0

    outputs = []
    output_grads = []
    for output_name in output_data:
        outputs.append(output_data[output_name])
        output_grads.append(torch.randn(output_data[output_name].size(), requires_grad=False, device=torch.cuda.current_device(), dtype=grad_type) )
    for output_extra_name in output_extra_tensors:
        outputs.append(output_extra_tensors[output_extra_name])
        output_grads.append(torch.randn(output_extra_tensors[output_extra_name].size(), requires_grad=False, device=torch.cuda.current_device(), dtype=grad_type) )

    if op_info["name"] == "encoder-embedding":
        input_tensors = None          
        torch.autograd.backward(outputs, grad_tensors=output_grads, retain_graph=True)                    
    else:
        input_tensors = []
        for input_name in input_data:
            input_tensors.append(input_data[input_name])
        for input_extra_name in input_extra_tensors:
            ## workaround for softmax op.
            if "mask" not in input_extra_name:
                input_tensors.append(input_extra_tensors[input_extra_name])
        torch.autograd.grad(outputs=outputs, grad_outputs=output_grads, inputs=input_tensors, allow_unused=False, retain_graph=True)

    mem_allocated_bwd = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    mem_reserved_bwd = torch.cuda.memory_reserved() / (1024.0 * 1024.0)

    _mem_reserved_bwd = mem_reserved_bwd - new_mem_reserved
    if _mem_reserved_bwd < 0:
        _mem_reserved_bwd = 0

    return avg_fwd_time, avg_bwd_time, _mem_reserved_fwd, _mem_reserved_bwd, _mem_allocated 

def dump_profiled_results(save_filename_prefix, mbs, algo, op_list):
    global profiled_results
    args = get_args()
    if torch.distributed.get_rank() == 0:
        print_rank0(f"====== PROFILING RESULTS ({save_filename_prefix}, mbs = {mbs}, tp = {args.prof_tp_size}, algo = {algo}) ======")
        save_file_name = f"{save_filename_prefix}_mbs{mbs}_tp{args.prof_tp_size}_algo{algo}.csv"
        result_title = ["op_name", "forward-compute", "backward-compute", "input_size", "output_size", "weights", "activations", "fwd_reserved", "bwd_reserved"] 

        f_result = open(args.prof_path + save_file_name,'w')
        f_csv = csv.writer(f_result)
        f_csv.writerow(result_title)
        for op_info in op_list:
            op_name = save_filename_prefix + op_info["name"] + f"mbs{mbs}tp_size{args.prof_tp_size}algo{algo}"
            fwd_time = '{:.3f}'.format(float(profiled_results[op_name][0]))
            bwd_time = '{:.3f}'.format(float(profiled_results[op_name][1]))            
            input_size = '{:.3f}'.format(float(profiled_results[op_name][2]))
            output_size = '{:.3f}'.format(float(profiled_results[op_name][3]))
            weight_size = '{:.3f}'.format(float(profiled_results[op_name][4]))
            activations = '{:.3f}'.format(float(profiled_results[op_name][5]))
            reserved_fwd = '{:.3f}'.format(float(profiled_results[op_name][6]))
            reserved_bwd = '{:.3f}'.format(float(profiled_results[op_name][7]))
            f_csv.writerow([op_info["name"], fwd_time, bwd_time, input_size, output_size, weight_size, activations, reserved_fwd, reserved_bwd]) 

        save_dict = {}
        save_dict["profiled_results"] = profiled_results
        save_dict["op_hash_list"] = op_hash_list
        pickle.dump(save_dict, open(args.prof_cache_file, "wb"))

def estimate_profile_time(task):
    global ref_data
    global new_hash_list
    model = task["model"]
    size = task["size"]
    mbs = task["mbs"]

    args = get_args()
    args.micro_batch_size = mbs
    op_list = get_model(model, size).full_op_list
    tp_size = args.prof_tp_size
    algo_list = model_prof_configs[model]["algo"]
    save_filename_prefix = f"{model}_{size}"
    
    sum_time = 0
    for algo in algo_list:
        for op_info in op_list:
            op_uniq_name = save_filename_prefix + op_info["name"] + f"mbs{mbs}tp_size{tp_size}algo{algo}" 
            op_hash = get_op_hash(op_info, mbs, tp_size, algo, save_filename_prefix)
            if op_uniq_name in ref_data:
                if op_hash not in new_hash_list:
                    _profiled_results = list(ref_data[op_uniq_name])
                    for _time in [_profiled_results[0], _profiled_results[1]]:
                        if _time < args.prof_warmup_threshold:
                            sum_time += _time * args.prof_warmup_times

                        if _time < args.prof_repeat_threshold:
                            sum_time += _time * args.prof_repeat_times[0]
                        else:
                            sum_time += _time * args.prof_repeat_times[1]
                    new_hash_list.append(op_hash)
                continue
            else:
                raise RuntimeError(f"op {op_uniq_name} not in database.")

    return sum_time / 1000000

def run_profile(task):
    global profiled_results
    model = task["model"]
    size = task["size"]
    mbs = task["mbs"]

    args = get_args()
    grad_type = torch.float
    op_list = get_model(model, size).full_op_list
    tp_size = args.prof_tp_size
    algo_list = model_prof_configs[model]["algo"]
    params_dtype = args.params_dtype
    save_filename_prefix = f"{model}_{size}"

    args.micro_batch_size = mbs
    for algo in algo_list:
        ## infer the data size according to op specs
        infer_data_size(op_list, save_filename_prefix, mbs, algo)
        # run profiling
        for op_info in op_list:
            op_uniq_name = save_filename_prefix + op_info["name"] + f"mbs{mbs}tp_size{tp_size}algo{algo}" 
            op_hash = get_op_hash(op_info, mbs, tp_size, algo, save_filename_prefix)
            if op_uniq_name in profiled_results:
                print_rank0(f"working on {op_info['name']}, mbs = {mbs}, tp = {tp_size}, algo = {algo} ... Hit same op in cache!!!")  
                continue
            elif op_hash in op_hash_list:
                print_rank0(f"working on {op_info['name']}, mbs = {mbs}, tp = {tp_size}, algo = {algo} ... Hit identical op in cache!!!")  
                _profiled_results = list(profiled_results[op_hash_list[op_hash]])
                _profiled_results[2] = input_size_dict[op_uniq_name]
                _profiled_results[3] = output_size_dict[op_uniq_name]
                profiled_results[op_uniq_name] = _profiled_results
                continue                             
            else:
                print_rank0(f"working on {op_info['name']}, mbs = {mbs}, tp = {tp_size}, algo = {algo} ... ")  
                try:
                    if SKIP_RUNNING:
                        _fwd_time, _bwd_time, _reserved_fwd, _reserved_bwd, _allocated_fwd = 0, 0, 0, 0, 0
                    else:
                        _fwd_time, _bwd_time, _reserved_fwd, _reserved_bwd, _allocated_fwd = profile_op(mbs, algo, op_info, params_dtype, grad_type, op_uniq_name) 
                except RuntimeError as e:
                    print(e)
                    _fwd_time, _bwd_time, _reserved_fwd, _reserved_bwd, _allocated_fwd = 10000000, 10000000, 10000000, 10000000, 10000000
                print(f"[results] {op_info['name']}: fwd_compute = {_fwd_time:.2f} us, bwd_compute = {_bwd_time:.2f} us, fwd_allocated = {_allocated_fwd:.1f} MB, fwd_reserved = {_reserved_fwd:.1f} MB, bwd_reserved = {_reserved_bwd:.1f} MB.")
            
            profiled_results[op_uniq_name] = [_fwd_time, _bwd_time, input_size_dict[op_uniq_name], output_size_dict[op_uniq_name], weight_size_dict[op_uniq_name], _allocated_fwd, _reserved_fwd, _reserved_bwd]
            op_hash_list[op_hash] = op_uniq_name
        dump_profiled_results(save_filename_prefix, mbs, algo, op_list)

def get_prof_tasks_by_rank(all_tasks, num_nodes, node_rank):
    '''
    This function is aimed to distribute the profiling tasks evenly to all the nodes, 
    to make each node have a similar profiling time.
    The distribution is achieved using the estimated profiling time of each task.
    We estimate the profiling time of each task using a given reference database.
    The reference database does not necessarily to be accurate, which can be a one-time 
    profiling database or database obtained in another environment.
    '''
    all_task_times = []
    for task in all_tasks:
        profile_time = estimate_profile_time(task)
        all_task_times.append(profile_time)
    sum_time = sum(all_task_times)
    range_per_rank = 1 / num_nodes
    _current_sum_time = 0
    _all_tasks = []
    print(f"[DEBUG] (tp_size = {args.prof_tp_size}). calculating prof tasks by rank ...")
    for i, profile_time in enumerate(all_task_times):
        _current_sum_time += profile_time 
        _current_ratio = _current_sum_time / sum_time
        _supposed_rank = int(max(0, _current_ratio - 0.01) / range_per_rank)
        if _supposed_rank == node_rank:
            _all_tasks.append(all_tasks[i])
            print(f"APPENDING task: {all_tasks[i]['model']}_{all_tasks[i]['size']}, mbs {all_tasks[i]['mbs']}. (accum time = {_current_sum_time:.2f} / {sum_time:.2f}, ratio = {_current_ratio:.2f})")
        else:
            print(f"DROP task: {all_tasks[i]['model']}_{all_tasks[i]['size']}, mbs {all_tasks[i]['mbs']}. (accum time = {_current_sum_time:.2f} / {sum_time:.2f}, ratio = {_current_ratio:.2f}). this should be profiled by rank {_supposed_rank}")
    return _all_tasks

if __name__ == "__main__":

    start_profiling_time = time.time()
    initialize_megatron()
    args = get_args()

    ## read cached database if exists
    if os.path.exists(args.prof_cache_file):
        cached_results = pickle.load(open(args.prof_cache_file, "rb"))
        profiled_results = cached_results["profiled_results"]
        op_hash_list = cached_results["op_hash_list"]
    else:
        profiled_results = {}
        op_hash_list = {}        

    ## get profiling tasks
    ## "task"s are defined by unique {model, size, mbs} pairs
    all_prof_tasks = []
    model_names = ["resnet", "gpt", "t5"] if args.prof_model_name == "all" else [args.prof_model_name]
    for model in model_names:
        model_sizes = model_prof_configs[model]["model_size"] if args.prof_model_size == "all" else [args.prof_model_size]
        for size in model_sizes:
            if args.prof_mbs_list is None:
                micro_batch_sizes = model_prof_configs[model]["mbs"]
            else:
                micro_batch_sizes = args.prof_mbs_list
            for mbs in micro_batch_sizes:
                all_prof_tasks.append({"model": model, "size": size, "mbs": mbs})

    ## distribute profiling tasks if using multiple nodes
    if args.prof_num_nodes is not None:
        new_hash_list = []
        ref_data = pickle.load(open(args.prof_ref_data, "rb"))["profiled_results"]
        all_prof_tasks = get_prof_tasks_by_rank(all_prof_tasks, args.prof_num_nodes, args.prof_node_rank)

    ## run profiling tasks
    for prof_task in all_prof_tasks:
        run_profile(prof_task)

    end_profiling_time = time.time()
    print_rank0(f"[TOTAL PROFILING TIME] {end_profiling_time - start_profiling_time:2f} s")
