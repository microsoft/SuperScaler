"""The entry point of intra-op + inter-op parallelism benchmark."""
import argparse
from datetime import datetime
import time

import numpy as np

from alpa.util import write_tsv, run_cmd, get_num_hosts_and_num_devices, to_str_round, GB

from benchmark_3d_one_case import benchmark_one_case
import suite_auto_gpt
import suite_auto_moe
import suite_manual_gpt
import suite_manual_moe
import suite_wresnet

import os 
import csv

RUNTIME_LOG_PATH = os.environ.get("RUNTIME_LOG_PATH", './') 
LOG_PATH = os.environ.get("LOG_PATH", './') 

benchmark_suites = {
    "gpt.tmp": suite_manual_gpt.tmp_suite,
    "gpt.tmp_auto": suite_auto_gpt.tmp_suite,
    "gpt.perf_test_manual": suite_manual_gpt.perf_test_suite,
    "gpt.perf_test_auto": suite_auto_gpt.perf_test_suite,
    "gpt.grid_search_auto": suite_auto_gpt.grid_search_suite,
    "gpt.correctness_test_auto": suite_auto_gpt.correctness_test_suite,
    "gpt.ae_small": suite_auto_gpt.ae_small_suite,
    "gpt.ae_large": suite_auto_gpt.ae_large_suite,
    "gpt.ae_scale_8": suite_auto_gpt.ae_scale_suite_8,
    "gpt.ae_scale_16": suite_auto_gpt.ae_scale_suite_16,
    "gpt.ae_scale_32": suite_auto_gpt.ae_scale_suite_32,
    "gpt.ae_scale_64": suite_auto_gpt.ae_scale_suite_64,
    "gpt.ae_scale_128": suite_auto_gpt.ae_scale_suite_128,

    "moe.tmp": suite_manual_moe.tmp_suite,
    "moe.tmp_auto": suite_auto_moe.tmp_suite,
    "moe.perf_test_auto": suite_auto_moe.perf_test_suite,
    "moe.grid_search_auto": suite_auto_moe.grid_search_suite,

    "wresnet.perf_test_auto": suite_wresnet.perf_test_auto_suite,
    "wresnet.grid_search_auto": suite_wresnet.grid_search_auto_suite,
    "wresnet.ae_small": suite_wresnet.ae_small_suite,
    "wresnet.ae_large": suite_wresnet.ae_large_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(benchmark_suites.keys()), type=str, required=True)
    parser.add_argument("--niter", type=int, default=3,
        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--no-separate-process", action='store_false',
                        help="Do not launch separate processes for benchmark."
                             "Errors in a single case will terminate this script.",
                        dest='use_separate_process')
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    # Get the benchmark suite
    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)
    num_gpus = num_hosts * num_devices_per_host

    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None
    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()
    run_cmd("mkdir -p tmp")

    model_type = args.suite.split(".")[0]
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_name = f"{model_type}_alpa_{args.exp_name}_{date_str}.tsv"

    start_time = time.time()
    iteration_time = 0
    # Run all cases
    for benchmark_case in suite:
        if model_type in ["gpt", "bert"]:
            (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
             num_micro_batches, parallel_mode, parallel_args) = benchmark_case
            model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads)
        elif model_type == "moe":
            (batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,
             num_experts, expert_group_size, num_micro_batches,
             parallel_mode, parallel_args) = benchmark_case
            model_config = (batch_size, seq_len, hidden_size, num_layers, num_heads, num_experts, expert_group_size)
        elif model_type == "wresnet":
            (batch_size, image_size, num_layers, num_channels, width_factor, dtype,
             num_micro_batches, parallel_mode, parallel_args) = benchmark_case
            model_config = (batch_size, image_size, num_layers, num_channels, width_factor)
        else:
            raise ValueError(f"Invalid model: {model_type}")

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(model_type, benchmark_case, args.niter,
                                    num_hosts, num_devices_per_host,
                                    use_separate_process=args.use_separate_process,
                                    disable_tqdm=args.disable_tqdm)
        (parameter_count, max_mem_allocated, latencies, tflops,
         tflops_ckpt, compilation_times, compute_cost_file_name, forward_stage_layer_ids,
         submesh_shapes, logical_mesh_shapes, autosharding_option_dicts) = result

        print(f"[DEBUG] latencies = {latencies}")

        heads = ["Type", "Model Config", "#Microbatch", "#GPU", "Parallel Config",
                 "Mean Time", "Std Time", "#Params", "TFLOPs",
                 "TFLOPs (ckpt)", "Peak Mem", "Compilation Time", "Global Batch Size"]
        values = [model_type, model_config, num_micro_batches, num_gpus, parallel_args,
                  f"{np.mean(latencies):.3f}s", f"{np.std(latencies):.3f}",
                  f"{parameter_count/1e9:.3f}B", f"{tflops:.2f}", f"{tflops_ckpt:.2f}",
                  f"{max_mem_allocated/GB:.3f}G", to_str_round(compilation_times, 2), batch_size]
        write_tsv(heads, values, RUNTIME_LOG_PATH + output_name)

        iteration_time += np.mean(latencies) * 3  # exclude the execution time
        time.sleep(0.1)  # for ctrl+c to work

    end_time = time.time()
    
    file_name = f'{LOG_PATH}{model_type}_alpa_{args.exp_name}_summary_{date_str}.csv'
    info_to_csv = [["search_cost"], [f"{(end_time - start_time - iteration_time):.2f}"]]
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)