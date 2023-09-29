# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time 
import os 
import json 

from aceso_cost_model import read_profiled_time, predict_time_breakdown
from model_ops_info import get_full_op_list
from aceso_utils import *

args = parse_args()
print_args(args)

MAX_VALUE = 2**30
MIN_VALUE = -2**30

mbs_list = args.micro_batch_size

full_op_list = get_full_op_list(args)
NUM_OPS = len(full_op_list)
read_profiled_time(args.model_name, args.model_size, args.profiled_time_path)

def get_balance_config(pp, tp, dp, base_mbs, recomp):
    num_ops_per_stage = [NUM_OPS]
    tp_per_op = [tp for _ in range(NUM_OPS)]
    dp_per_op = [dp for _ in range(NUM_OPS)]
    if recomp:
        recompute_ops = [1 for _ in range(NUM_OPS)]
    else:
        recompute_ops = [0 for _ in range(NUM_OPS)]
    if args.model_name in ["gpt", "t5", "scale-layer"]:
        recompute_ops[0] = 0
        recompute_ops[-1] = 0
    algo_list = [0 for _ in range(NUM_OPS)]
    base_batch_size = base_mbs

    if args.model_name in ["gpt", "scale-layer"]:
        config_list = []
        if args.num_layers % pp == 0:
            num_ops_list = [(args.num_layers//pp) * 13 for _ in range(pp)]
            num_ops_list[0] += 1
            num_ops_list[-1] += 2
            config_list.append(get_config(num_ops_list, tp_per_op, dp_per_op, recompute_ops, base_batch_size, args.global_batch_size, full_op_list, algo_list))
        return config_list
    elif args.model_name == "t5":
        config_list = []
        if pp == 1:
            num_ops_list = [args.num_layers * (13 + 21) + 5]
            config_list.append(get_config(num_ops_list, tp_per_op, dp_per_op, recompute_ops, base_batch_size, args.global_batch_size, full_op_list, algo_list))
        else:
            for pp_split_rank in range(1, pp):
                if args.num_layers % pp_split_rank == 0 and args.num_layers % (pp - pp_split_rank) == 0:
                    num_ops_list = [(args.num_layers//pp_split_rank) * 13 for _ in range(pp_split_rank)]
                    num_ops_list[0] += 1
                    num_ops_list[-1] += 1
                    num_ops_list += [(args.num_layers//(pp - pp_split_rank)) * 21 for _ in range(pp - pp_split_rank)]
                    num_ops_list[pp_split_rank] += 1
                    num_ops_list[-1] += 2
                    config_list.append(get_config(num_ops_list, tp_per_op, dp_per_op, recompute_ops, base_batch_size, args.global_batch_size, full_op_list, algo_list))
        return config_list
    elif args.model_name == "resnet":
        num_ops_list = [(args.num_layers//pp) * 8 for _ in range(pp)]
        num_ops_list[0] += 4
        num_ops_list[-1] += 2

        num_ops_list[-1] += (args.num_layers - (args.num_layers//pp) * pp) * 8
        config_list = []
        config_list.append(get_config(num_ops_list, tp_per_op, dp_per_op, recompute_ops, base_batch_size, args.global_batch_size, full_op_list, algo_list))
        return config_list

start_time = time.time()

if (args.model_name == "gpt" and args.model_size == "350M") or (args.model_name == "t5" and args.model_size == "220M"):
    max_tp_size = min(args.max_tp, 4)
else:
    max_tp_size = min(args.max_tp, 8)

tp_size_list = []
tp = 1
while tp <= max_tp_size:
    tp_size_list.append(tp)
    tp *= 2

all_available_configs = []
for num_stages in range(args.start_num_stages, args.end_num_stages+1):
    for tp_size in tp_size_list:
        if args.num_gpus % (num_stages*tp_size) == 0:
            dp_size = args.num_gpus // (num_stages*tp_size)
            for base_mbs in mbs_list:  
                if base_mbs // dp_size not in mbs_list:
                    continue        
                for recomp in [True, False]:
                    print(f"working on num_stages[{num_stages}], tp_size[{tp_size}], dp_size[{dp_size}], mbs[{base_mbs}], recomp[{recomp}] ...")
                    current_configs = get_balance_config(num_stages, tp_size, dp_size, base_mbs, recomp)
                    if current_configs is not None:
                        all_available_configs += current_configs
                    else:
                        print(f"config is None.")

min_time = MAX_VALUE
best_config = None
num_feasible_cases = 0
all_feasible_configs = {}
all_feasible_time = []
for current_config in all_available_configs:
    predict_time_breakdown(current_config)
    current_time = max(current_config.time_list)
    current_memory = max(current_config.memory_list)

    if current_memory < args.memory_limit:
        print_simple_config_info(current_config, info=f"case {num_feasible_cases}", print_debug_info=args.print_debug_info)
        if current_time not in all_feasible_time:
            all_feasible_time.append(current_time)
        if current_time not in all_feasible_configs:
            all_feasible_configs[current_time] = current_config
        else:
            if current_memory < max(all_feasible_configs[current_time].memory_list):
                all_feasible_configs[current_time] = current_config

        if current_time < min_time:
            min_time = current_time
            best_config = current_config
        num_feasible_cases += 1
    else:
        print_simple_config_info(current_config, info=f"fail case", print_debug_info=args.print_debug_info)

end_time = time.time()
print("\n")
if num_feasible_cases > 0:
    all_feasible_time.sort()
    for i in range(min(args.num_of_saved_configs, len(all_feasible_time))):
        feasible_time = all_feasible_time[i]
        print_simple_config_info(all_feasible_configs[feasible_time], info=f"best_{i}", print_debug_info=args.print_debug_info)
        dump_config_to_json(all_feasible_configs[feasible_time], f"{args.config_save_path}{args.model_name}_{args.model_size}_{args.num_layers}layers_megatron_est_{int(feasible_time)}.json", args)
else:
    print("no feasible config")

print(f"[TOTAL TIME] {end_time - start_time} s.")