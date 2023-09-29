# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
op_list = None
full_op_list = None
tunable_op_list = None
no_recompute_op_list = None

def get_op_list(args):
    global op_list
    if op_list is None:
        op_list = []
        if args.model_name == "scale-layer":
            src_data_file = args.profiled_time_path + f"gpt_scale-layer_mbs{args.micro_batch_size[0]}_tp1_algo0.csv"
        else:
            model_size = args.model_size
            if args.model_name == "t5" and args.model_size == "22B":
                model_size = "11B"
            src_data_file = args.profiled_time_path + args.model_name + f"_{model_size}_mbs{args.micro_batch_size[0]}_tp1_algo0.csv"
        with open(src_data_file) as f:
            src_data = csv.reader(f)
            line_index = 0
            for row in src_data:
                line_index += 1
                if line_index > 1:
                    op_list.append(row[0])   
    return op_list   

def get_full_op_list(args):
    global op_list, full_op_list
    if op_list is None:
        op_list = get_op_list(args)
    if full_op_list is None:
        if args.model_name in ["gpt", "scale-layer"]:
            head_ops = [op_list[0]]
            decoder_layer = op_list[1:14]
            tail_ops = op_list[14:]
            full_op_list = head_ops + decoder_layer * args.num_layers + tail_ops
        elif args.model_name in ["t5"]:
            encoder_head_ops = [op_list[0]]
            encoder_layer = op_list[1:14]
            encoder_tail_ops = [op_list[14]]
            decoder_head_ops = [op_list[15]]
            decoder_layer = op_list[16:37]
            decoder_tail_ops = op_list[37:]
            full_op_list = encoder_head_ops + encoder_layer * args.num_layers + encoder_tail_ops + \
                decoder_head_ops + decoder_layer * args.num_layers + decoder_tail_ops
        elif args.model_name in ["resnet"]:
            full_op_list = op_list
        else:
            raise RuntimeError(f"model {args.model_name} not supported yet.")
    return full_op_list

def get_tunable_op_list(args):
    """
    Tunable operators are the oprators which support tensor parallelims, we can tune the tensor-parallelism size.
    """
    global tunable_op_list, op_list
    if tunable_op_list is None:
        if args.model_name in ["gpt", "scale-layer"]:
            tunable_op_list = ["encoder-embedding", "enc-attention-qkv", "enc-attention-dense", "enc-MLP-GEMM-1", "enc-MLP-GEMM-2"]
        elif args.model_name in ["t5"]:
            tunable_op_list = ["encoder-embedding", "enc-attention-qkv", "enc-attention-dense", "enc-MLP-GEMM-1", "enc-MLP-GEMM-2", "dec-attention-qkv-1", "dec-attention-dense-1", "dec-attention-qkv-2","dec-attention-dense-2", "dec-MLP-GEMM-1", "dec-MLP-GEMM-2"]
        elif args.model_name in ["resnet"]:
            if op_list is None:
                op_list = get_op_list(args)
            tunable_op_list = []
            for op in op_list:
                if "conv" in op or "downsample" in op:
                    tunable_op_list.append(op)            
        else:
            raise RuntimeError(f"model {args.model_name} not supported yet.")
    return tunable_op_list

def get_no_recompute_op_list(args):
    """
    Operators that should not be recomputed.
    """
    global no_recompute_op_list
    if no_recompute_op_list is None:
        if args.model_name in ["gpt", "scale-layer"]:
            no_recompute_op_list = ["encoder-embedding", "gpt-post-process"]
        elif args.model_name in ["t5"]:
            no_recompute_op_list = ["encoder-embedding", "decoder-embedding", "t5-post-process"]
        elif args.model_name in ["resnet"]:
            no_recompute_op_list = ["conv1"]
        else:
            raise RuntimeError(f"model {args.model_name} not supported yet.")
    return no_recompute_op_list

def get_op_spec(op_name, tp_size, dp_size, algo_index, input_spec):
    if op_name in ["encoder-embedding"]:
        if input_spec:
            return None
        else:
            return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
    elif op_name in ["enc-1st-layernorm", "enc-2nd-layernorm"]:
        return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
    elif op_name in ["final-layernorm"]:
        if input_spec:
            return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}        
        else:
            return {"R": tp_size, "V": 1, "dims": [dp_size, 1, 1]}
    elif op_name in ["enc-attention-qkv", "enc-MLP-GEMM-1"]:
        if algo_index == 0: # column
            if input_spec:
                return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
            else:
                return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
        elif algo_index == 1: # row
            if input_spec:
                return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
            else:
                return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
    elif op_name in ["enc-attention-score"]:
        if input_spec:
            return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
        else:
            return {"R": 1, "V": 1, "dims": [dp_size, tp_size, 1]}      
    elif op_name in ["enc-attention-softmax", "enc-attention-dropout"]:
        return {"R": 1, "V": 1, "dims": [dp_size, tp_size, 1]}
    elif op_name in ["enc-attention-context"]:
        if input_spec:
            return {"R": 1, "V": 1, "dims": [dp_size, tp_size, 1]}
        else:
            return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
    elif op_name in ["enc-attention-dense", "enc-MLP-GEMM-2"]:
        if algo_index == 0: # row
            if input_spec:
                return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
            else:
                return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
        elif algo_index == 1: # column
            if input_spec:
                return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
            else:
                return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
    elif op_name in ["enc-post-attention-dropout", "enc-post-MLP-dropout"]:
        return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
    elif op_name in ["enc-MLP-gelu"]:
        return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
    elif op_name in ["gpt-post-process"]:
        if input_spec:
            return {"R": tp_size, "V": 1, "dims": [dp_size, 1, 1]}
        else:
            return None
    ## resnet operators:
    elif "conv" in op_name or "downsample" in op_name or "bn" in op_name or "relu" in op_name or "pool" in op_name or "fc" in op_name:
        return {"R": tp_size, "V": 1, "dims": [dp_size, 1, 1]}



    
