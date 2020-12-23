# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import subprocess
import argparse
from superscaler.scaler_graph import DataParallelism
from training import cifarnet
import superscaler.tensorflow as superscaler


def is_gpu_available():
    """
        Check NVIDIA with nvidia-smi command
        Returning code == 0 and count > 0, it means NVIDIA is installed
        and GPU is available for running
        Other means not installed
    """
    code = os.system('nvidia-smi')
    if code == 0:
        cmd = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
        count = subprocess.check_output(cmd, shell=True)
        return int(count) > 0
    else:
        return False


def test_train_cifarnet():
    # Create Superscaler_TF class
    sc = superscaler()

    # Init parameters
    session_run_params = cifarnet.model()
    dataset_paths = cifarnet.get_dataset_paths()
    strategy = DataParallelism(range(2))
    deployment_setting = {"1": "localhost"}
    communication_DSL = "ring"
    resource_pool = os.path.join(os.path.dirname(__file__), 'plan_gen', 'data',
                                 'resource_pool.yaml')

    # Init runtime argument
    parser = argparse.ArgumentParser(description='Test Runner')

    args, _ = parser.parse_known_args()
    args.steps = 2
    args.interval = 1
    args.print_info = True
    args.print_fetches_targets = True

    if is_gpu_available():
        # Init Superscaler_TF class
        sc.init(session_run_params, deployment_setting, strategy,
                communication_DSL, resource_pool, dataset_paths)
        cache_dir = sc.get_cache_dir()
        # Check whether cache_dir is created
        if not os.path.exists(cache_dir):
            raise OSError
        # Check whether working_dir is sub_folder of cache_dir
        assert (os.path.samefile(cache_dir,
                                 os.path.dirname(sc.get_working_dir())))
        if not os.path.exists(sc.get_working_dir()):
            raise OSError

        for i in range(sc._graph_count):
            working_dir = sc.get_working_dir()
            tmp_rank_dir = os.path.join(working_dir, str(i))
            if not os.path.exists(tmp_rank_dir):
                raise OSError

            plan_path = os.path.join(tmp_rank_dir, 'plan.json')
            plan_ref = json.load(open(plan_path, 'r'))
            assert (plan_ref == sc._assigned_plan[i])

            model_desc_path = os.path.join(tmp_rank_dir, 'model_desc.json')
            model_desc_ref = json.load(open(model_desc_path, 'r'))
            assert (model_desc_ref == sc._graph_config)

            graph_path = os.path.join(tmp_rank_dir, 'graph.pbtxt')
            graph_ref = open(graph_path, 'r').read()
            assert (graph_ref == sc._partition_graphs[i])

        # final run
        args.steps = 2
        args.interval = 1
        args.print_info = True
        args.print_fetches_targets = True
        sc.run(args)
