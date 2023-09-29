# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as multiproc
import time
import csv
import torch.nn.functional as F
result_file_name = os.environ.get("FILE_NAME", "p2p_band.log")

def run(local_rank, global_rank):
    """ Simple collective communication. """
    global result_file_name
    all_data_sizes = []
    all_bandwidths = []
    warmup_times = 20
    repeat_times = 50
    torch.cuda.set_device(local_rank)

    for i in range(11):
        data_size_in_mb = 2**i
        all_data_sizes.append(data_size_in_mb)
        data_size = data_size_in_mb * 1024 * 1024 // 2
        tensor = torch.ones(data_size, dtype=torch.float16).cuda()

        if global_rank == 0:
            for i in range(warmup_times): 
                dist.send(tensor, dst=1-global_rank)
            time_list = []
            for i in range(repeat_times): 
                torch.cuda.synchronize()
                start = time.time()          

                dist.send(tensor, dst=1-global_rank)
                torch.cuda.synchronize()
                end = time.time()    
                time_list.append((end - start)*1000)    

        elif global_rank == 1:
            for i in range(warmup_times): 
                dist.recv(tensor, src=1-global_rank)
            time_list = []
            for i in range(repeat_times): 
                torch.cuda.synchronize()
                start = time.time()          
                dist.recv(tensor, src=1-global_rank)
                torch.cuda.synchronize()
                end = time.time()    
                time_list.append((end - start)*1000)    

        avg_time_result_in_ms = sum(time_list)/repeat_times
        bandwidth_in_gb_per_second = (data_size_in_mb/1024) / (avg_time_result_in_ms/1000)
        all_bandwidths.append(f"{bandwidth_in_gb_per_second:.2f}")
        result_string = f'Rank {global_rank} | Time(averaged {repeat_times} times) = {avg_time_result_in_ms:.2f} ms, data_size = {data_size_in_mb:.2f} MB, bandwidth = {bandwidth_in_gb_per_second:.2f} GB/s'
        print(result_string)
    if global_rank == 0:
        with open(result_file_name, "a+") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(all_data_sizes)
            f_csv.writerow(all_bandwidths)

def init_process(local_rank, global_rank, world_size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000') 
    init_method += master_ip + ':' + master_port

    dist.init_process_group(
        backend=backend,
        world_size=world_size, rank= global_rank,
        init_method=init_method)
    
    fn(local_rank, global_rank)

if __name__ == "__main__":
    gpus_per_node = int(os.getenv('GPUS_PER_NODE', 1))
    num_nodes = int(os.getenv('NNODES', 1))
    node_rank = int(os.getenv('NODE_RANK', 0))
    multiproc.set_start_method("spawn")
    world_size = gpus_per_node * num_nodes

    processes = []
    for local_rank in range(gpus_per_node):
        global_rank = gpus_per_node * node_rank + local_rank
        p = multiproc.Process(target=init_process, args=(local_rank, global_rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
