# Aceso Artifact Instructions

## Overall Workflow
The workflow of Aceso is:
- Step 1: **profile** basic information (e.g., op computation time and communication time).
- Step 2: **search** for the best configs using the profiled database.
- Step 3: **train** the model with the found config.

In this artifact evaluation, we will check the functionality of Aceso, by working through the full process (profile + search + train) with a small setup (4GPUs). And we encourage you to perform part of the large-scale experiments (only the search step) in the Aceso paper because the search step does not require GPUs and we will provide our profiled database.

## Set up the environment

- **Pull the submodules:**
    ```
    git submodule update --init --recursive
    ```
- **Patch Megatron-LM:**
    ```
    cd external/Megatron-LM
    cp ../aceso_ae_megatron.patch ./
    git apply --whitespace=nowarn aceso_ae_megatron.patch
    ```
- **Install dependencies:**
    - **Option 1: docker (recommended)**

        We provide a `Dockerfile` to build the needed docker image, you can build the docker by executing:
        ```
        docker build -t aceso_image:latest .
        ```
        After image built, launch a container with the docker image on each node in your cluster: (replace `aceso_path` with the path of this repo in your system)
        ```
        docker run -it -d --name=aceso-ae --net=host --ipc=host --gpus=all -v aceso_path:aceso_path aceso-image bash
        ```
        Without specifically noted, all the following scripts are executed inside docker container.

    - **Option 2: manual installation**
        - Python >=3.7
        - CUDA 11.6
        - PyTorch 1.12.0:
            ```
            pip3 install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch/
            ```
        - Dependencies of Aceso runtime & Megatron-LM:
            ```
            pip3 install -r requirements.txt
            ```
        - Apex:
            ```
            cd external/apex
            pip3 install -r requirements.txt
            pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  ./
            ```
        - Alpa 0.1.5 (Baseline system): 
            ```
            cd external/alpa
            pip3 install -r requirements.txt
            pip3 install https://github.com/alpa-projects/alpa/releases/download/v0.2.3/jaxlib-0.3.5%2Bcuda113.cudnn820-cp37-none-manylinux2010_x86_64.whl
            sudo apt-get update && apt install coinor-cbc -y
            ```
        - Parallel-ssh: (for distributed training)
            ```
            sudo apt-get -y install pssh
            ```

## Functionality check with small-scale experiments (4 hours)
**Hardware requirement:** 4 GPUs. 

The models chosen for the small-scale experiments: GPT-3(1.3B), T5(770M), and Wide-ResNet(1B). The global batch size is set to 512 for GPT-3 and T5 models and 768 for Wide-ResNet.

In this small-scale experiment, we will run the search & train steps of Aceso and the two baselines (Alpa and Megatron-LM), at last we will compare the training throughput and search cost. We will also check the prediction accuracy of Aceso's performance model. 

You can follow the instructions to perform the experiments step by step. And you can also execute only one script, which contains all the steps:
```
bash scripts/run_all_small.sh
```

### (Optional) Step 1: Profile (40 minutes)
The profile step can be skipped in the artifact as we provided a pre-profiled database `profiler/profiled-time-miniset/` for the small-scale experiment. But you can also profile on your own by executing:  
```
cd profiler
bash scripts/profile_small.sh
```

### Step 2: Search (6 minutes)
Run the search for GPT-3(1.3B), T5(770M) and Wide-ResNet(1B) models:
- GPT-3 (2 minutes): `bash scripts/aceso_gpt_search.sh small`
- Wide-ResNet (1 minute): `bash scripts/aceso_resnet_search.sh small`
- T5 (3 minutes): `bash scripts/aceso_t5_search.sh small`

### Step 3: Train (6 minutes)
We will train each model with the found configurations for 3 iterations to get the iteration time:

- GPT-3 (3 minutes): `bash scripts/aceso_gpt_execute.sh small`
- Wide-ResNet (1 minute): `bash scripts/aceso_resnet_execute.sh small`
- T5 (2 minutes): `bash scripts/aceso_t5_execute.sh small`

**Example output:**
```
-------- gpt End-to-end throughput --------
Size	 Batch Size	 Time(s)	 Thpt(samples/s)
1_3B	 512.0		 58.98		 8.68
```
### Step 4: Compare with Alpa & Megatron-LM (3 hours)
- **(Optional) Profile database for Alpa (30 minutes)**
    The profile step can be skipped in the artifact as we provided a pre-profiled database for Alpa (`external/alpa/prof_database.pkl`). You can also profile on your own by executing: 
    ```
    cd external/alpa
    python3 gen_prof_database.py --max-comm-size-intra-node 32 --max-comm-size-inter-node 29
    ```
- **Run Alpa (Search & Train): (3 hours)**
    - GPT-3 (1 hour): `bash scripts/alpa_gpt_search_execute.sh small`
    - Wide-ResNet (2 hours): `bash scripts/alpa_wresnet_search_execute.sh small`
- **Run Megatron-LM (Search & Train): (9 minutes)**
    - GPT-3 (4 minutes): `bash scripts/megatron_gpt_search.sh small && bash scripts/megatron_gpt_execute.sh small`
    - Wide-ResNet (2 minutes): `bash scripts/megatron_resnet_search.sh small && bash scripts/megatron_resnet_execute.sh small`
    - T5 (3 minutes): `bash scripts/megatron_t5_search.sh small && bash scripts/megatron_t5_execute.sh small`
- **Compare training throughput:**
    ```
    python3 scripts/get_e2e_performance.py small
    ```
    **Expected results:**
    ```
    -------- [gpt] End-to-end Throughput (Samples/s) --------
    Size     Megatron-LM     Alpa    Aceso
    1_3B    6.98             7.46    8.68
    -------- [t5] End-to-end Throughput (Samples/s) --------
    Size     Megatron-LM     Alpa    Aceso
    770M    9.94             -       15.14
    -------- [resnet] End-to-end Throughput (Samples/s) --------
    Size     Megatron-LM     Alpa    Aceso
    1B      34.59            36.00   41.52
    ```
- **Compare search cost:**
    ```
    python3 scripts/get_search_cost.py small
    ```
    **Expected results:**
    ```
    -------- [gpt] Search Cost (s) --------
    Size     Alpa    Aceso
    1_3B    4092.34  99.31
    -------- [t5] Search Cost (s) --------
    Size     Alpa    Aceso
    770M    -        200.10
    -------- [resnet] Search Cost (s) --------
    Size     Alpa    Aceso
    1B      5780.60  45.49
    ```
### Step 5: Check performance model accuracy
```
python3 scripts/get_perf_model_acc.py small
```
**Expected results:**
```
-------- [gpt] Time Prediction (s) --------
Size     Actual          Predict
1_3B    59011.61         58760.92
-------- [gpt] Memory Prediction (MB) --------
Size     Actual          Predict (normal + extra)
1_3B    10692.00         11940.64 (10660.64 + 1280.00)
-------- [t5] Time Prediction (s) --------
Size     Actual          Predict
770M    33823.79         33659.50
-------- [t5] Memory Prediction (MB) --------
Size     Actual          Predict (normal + extra)
770M    10470.00         11981.22 (11319.47 + 661.75)
-------- [resnet] Time Prediction (s) --------
Size     Actual          Predict
1B      18662.83         17672.05
-------- [resnet] Memory Prediction (MB) --------
Size     Actual          Predict (normal + extra)
1B      14060.00         11866.20 (10334.83 + 1531.38)
```

## Reproducing results with large-scale experiments
**Hardware requirement:** The full evaluation conducted in the paper requires 4 nodes with 8 V100(32GB) GPUs in each. But you can still check part of the results described in the paper because the **search** step does not require GPUs. 

### (Optional) Step 1: Profile (2 hours)
**Hardware requirement:** 32 GPUs (4 nodes * 8 GPUs/node).

The profile step can be skipped in the artifact as we provided a pre-profiled database `profiler/profiled-time-eurosys/` for the large-scale experiment. But you can also profile on your own: (all the profiled results will be saved into `profiler/profiled-time-eurosys-new/` by default)

- **Prepare for distributed profiling:** 
    - Modify the `profiler/profile_large_[gpt,t5,resnet].sh` file, edit `NUM_NODES` and `NODE_RANK` for each node.
    - Modify the `profiler/profile_large_dist_p2p.sh` file, edit `MASTER_ADDR` and `NODE_RANK` for each node. 
- **Parallel profiling with multiple nodes:** (execute this command in each node)
    ```
    cd profiler
    bash scripts/profile_large.sh
    ```
- **Gather the profiled results** into one directory `profiler/profiled-time-eurosys-new/` and share the directory among all the nodes.

### Step 2: Search (45 minutes)
**Hardware requirement:** CPU only.
The following script will run the search of all the model sizes considered in the paper:
- GPT-3 (15 minutes): `bash scripts/aceso_gpt_search.sh large`
- Wide-ResNet (15 minute): `bash scripts/aceso_resnet_search.sh large`
- T5 (15 minutes): `bash scripts/aceso_t5_search.sh large`

All the found configs will be saved into `logs/aceso/configs/[model_name]/[model_size]/top_configs/` as `.json` files. We have shown two case studies in our paper, about the config of GPT-3 1.3B and Wide-ResNet 6.8B, in Sec 5.4. You can check the found configs and compare them with the ones in case studies.

### Step 3: Train (2hours)
**Hardware requirement:** 32 GPUs (4 nodes * 8 GPUs/node).

- **Prepare for distributed training:** 
    - **Modify the training scripts:** On each node, edit the file `runtime/scripts/run_[gpt,t5,resnet]_[2nodes,4nodes].sh`(if using docker, edit `runtime/scripts/run_[gpt,t5,resnet]_[2nodes,4nodes]_docker.sh`): modify `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` with the information of your cluster.
    - **Create parallel-ssh host file:** In the `runtime` path, create a file named `pssh-2nodes.host`, which contains the two nodes you want to use in 2-node distributed training, e.g.,:
        ```
        10.0.0.1
        10.0.0.2
        ```
        Similarly, create  a `pssh-4nodes.host` file.
    - If using docker, please make sure the container is named as `aceso-ae` in all the nodes.

- **Train: (2 hours)**
    Train all the configs found in the search step: 
    - **If you are NOT using docker:**
        - GPT-3 (35 minutes): `bash scripts/aceso_gpt_execute.sh large`
        - Wide-ResNet (25 minutes): `bash scripts/aceso_resnet_execute.sh large`
        - T5 (1 hour): `bash scripts/aceso_t5_execute.sh large`
    - **If you are using docker: (execute outside of docker container)**
        - GPT-3 (35 minutes): `bash scripts/aceso_gpt_execute_docker.sh large`
        - Wide-ResNet (25 minutes): `bash scripts/aceso_resnet_execute_docker.sh large`
        - T5 (1 hour): `bash scripts/aceso_t5_execute_docker.sh large`

### Step 4: Compare with Alpa & Megatron-LM (29 hours)
**Hardware requirement:** 32 GPUs (4 nodes * 8 GPUs/node).

- **Alpa (27 hours):**
    - **Prepare for distributed training:** Launch a [Ray](https://docs.ray.io/en/latest/index.html) cluster. 
    - **Run Alpa (Search & Train): (27 hours)**
        - GPT-3 (9 hours): `bash scripts/alpa_gpt_search_execute.sh large`
        - Wide-ResNet (18 hours): `bash scripts/alpa_wresnet_search_execute.sh large`
- **Megatron-LM (2 hours):**
    - **Prepare for distributed training:** Similar to Aceso, modify `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` in the file `scripts/megatron_dist_scripts/run_[gpt,t5,resnet]_[2nodes,4nodes].sh` (if using docker, edit `scripts/megatron_dist_scripts/run_[gpt,t5,resnet]_[2nodes,4nodes]_docker.sh` instead). 
    - **Search:**
        - GPT-3: `bash scripts/megatron_gpt_search.sh large`
        - Wide-ResNet: `bash scripts/megatron_resnet_search.sh large`
        - T5: `bash scripts/megatron_t5_search.sh large`
    - **Train:**
        - **If you are NOT using docker:**
            - GPT-3 (30 minutes): `bash scripts/megatron_gpt_execute.sh large`
            - Wide-ResNet (20 minutes): `bash scripts/megatron_resnet_execute.sh large`
            - T5 (1 hour): `bash scripts/megatron_t5_execute.sh large`
        - **If you are using docker:**
            - GPT-3 (30 minutes): `bash scripts/megatron_gpt_execute_docker.sh large`
            - Wide-ResNet (20 minutes): `bash scripts/megatron_resnet_execute_docker.sh large`
            - T5 (1 hour): `bash scripts/megatron_t5_execute_docker.sh large`
 - **Compare training throughput: (Figure 7)**
    ```
    python3 scripts/get_e2e_performance.py large
    bash scripts/plot_fig7.sh
    ```
- **Compare search cost: (Figure 8)**
    ```
    python3 scripts/get_search_cost.py large
    bash scripts/plot_fig8.sh
    ```

### Scale to 1K layers (7.5 hours)
**Hardware requirement:** 8 GPUs (1 node * 8 GPUs/node)

In this experiment, we will run the search and train step of a customized GPT model, scaling the number of layers from 8 to 1024.

- **Aceso:(1.5 hours)**
    ```
    bash scripts/aceso_gpt_search.sh scale
    bash scripts/aceso_gpt_execute.sh scale
    ```
- **Alpa: (5 hours)**
    ```
    ray start --head
    bash scripts/alpa_gpt_search_execute.sh scale
    ray stop
    ```
- **Megatron-LM: (1 hour)**
    ```
    bash scripts/megatron_gpt_search.sh scale
    bash scripts/megatron_gpt_execute.sh scale
    ```
- **Compare training throughput and search cost: (Figure 9)**
    ```
    python3 scripts/get_e2e_performance.py scale
    python3 scripts/get_search_cost.py scale
    bash scripts/plot_fig9.sh
    ```

## Detailed usage of Aceso
This section is a reference on the detailed usage of each Aceso component: profiler, search algorithm, and runtime. 
### Profiling 
**Profile op-related information:** (e.g., forward/backward execution time, input/output size, weight size, and reserved memory size)
```
## In the `Aceso/profiler` path
python3 op_profiler.py \
    --prof-tp-size 1 \
    --prof-path PATH_TO_RESULT \
    --prof-cache-file PATH_TO_CACHE_FILE \
    --prof-model-name gpt \
    --prof-model-size all \
    --prof-repeat-times 40 10 \
    --prof-repeat-threshold 5000 \
    --prof-warmup-times 10 \
    --prof-warmup-threshold 100000
```
**Arguments:**
- `--prof-tp-size`: tensor-parallelism size 
- `--prof-path`: path to store the results
- `--prof-cache-file`: path to the cache file. 
    - If the path exists, use the cached results to speed up the profiling, otherwise cache the profiling results in the path.
- `--prof-model-name`: one of [`gpt`, `t5`, `resnet`]
- `--prof-model-size`: 
    - for `gpt`: one of ["350M", "1_3B", "2_6B", "6_7B", "13B"]
    - for `t5`: one of ["770M", "3B", "6B", "11B"]
    - for `resnet`: one of ["500M", "2B", "4B", "6_8B", "13B"]
    - You can also specify `all` to profile all the model sizes in the list. 
- `--prof-repeat-times`: number of repeat times for each operator. 
    - This argument requires two numbers in the form "num1 num2", which represents the repeat times for smaller operators and larger operators. 
- `--prof-repeat-threshold`: the boundary execution time (us) of smaller and larger operators.
- `--prof-warmup-times`: number of warmup times for small operators. 
- `--prof-warmup-threshold`: the boundary execution time (us) of smaller and larger operators.

**Profile collective communication time:**
```
## In the `Aceso/profiler` path
python3 comm_profiler.py \
    --prof-path PATH_TO_RESULT \
    --prof-cache-file PATH_TO_CACHE_FILE \
    --prof-op-time-path PATH_TO_OP_PROFILING_RESULT \
    --prof-tp-size 1 \
    --prof-model-name gpt \
    --prof-model-size all \
    --prof-warmup-time 10 \
    --prof-repeat-time 40
```
**Arguments:**
- `--prof-path`: path to store the results.
- `--prof-cache-file`: path to the cache file. 
    - If the path exists, use the cached results to speed up the profiling, otherwise cache the profiling results in the path.
- `--prof-op-time-path`: path to op-profiling results, which contains all the data sizes that need to be profiled.
- `--prof-tp-size`: collective communication world size.
- `--prof-model-name` and `--prof-model-size` are the same as op-profiling command.
- `--prof-warmup-time`: warmup times for each collective.
- `--prof-repeat-time`: repeat times for each collective.

All the profiled results will be saved as `.csv` files under the given `--prof-path`. 

### Search
Search for the best configs given model information, hardware information, profiled database, and search-related hyper-parameters:
```
## In the `Aceso/search` path
python3 aceso_search.py \
    --model-name gpt \
    --model-size 1_3B \
    --global-batch-size 1024 \
    --micro-batch-size 1 2 4 8 \
    --num-nodes 1 \
    --num-gpus-per-node 4 \
    --memory-limit 12000 \
    --log-path PATH_TO_LOGS \
    --profiled-time-path PATH_TO_PROFILING_RESULT \
    --config-save-path PATH_TO_SAVE_CONFIGS \
    --config-suffix UNIQUE_CONFIG_SUFFIX \
    --max-num-hops 7 \
    --time-budget-total 200
```
- **Model information:**
    - `--model-name` and `--model-size`: same as the arguments above for profiling.
    - `--global-batch-size`: global batch size.
    - `--micro-batch-size`: a list of considered micro batch size.
- **Hardware information:**
    - `--num-nodes`: number of nodes.
    - `--num-gpus-per-node`: number of GPUs in each node.
    - `--memory-limit`: memory limit in each GPU.
- **Profiled database:**
    - `--profiled-time-path`: path of the profiled database.
- **Search-related hyper-parameters:**
    - `--max-num-hops`: maximum number of search hops (depth).
    - `--time-budget-total`: budget of search.
- **Other arguments:**
    - `--log-path`: path of logs.
    - `--config-save-path`: path to save the found configs.
    - `--config-suffix`: a unique id for the configs.

All the found configs will be saved into `--config-save-path` as `.json` files. Here is one example config:
```
{
    "model_name": "gpt",
    "model_size": "1_3B",
    "num_layers": 24,
    "seq_length": 2048,
    "max_position_embeddings": 2048,
    "num_attention_heads": 32,
    "hidden_size": 2048,
    "global_batch_size": 32,
    "micro_batch_size": 1,
    "num_stages": 3,
    "num_gpus": [1, 1, 2],
    "checkpoint_activations": [true, true, false],
    "resharding_stages": [false, false, false],
    "num_ops_in_each_stage": [85, 91, 139],
    "model_parallel_size_of_each_op": [
        [1, 1, 1, .... 1], [1, 1, 1, .... 1], [2, 2, 2, .... 2]
    ],
    "data_parallel_size_of_each_op": [
        [1, 1, 1, .... 1], [1, 1, 1, .... 1], [1, 1, 1, .... 1]
    ],
    "recompute_ops": [
        [0, 0, 0, .... 0], [0, 0, 0, .... 0], [0, 0, 0, .... 0]
    ],
    "algo_of_each_op": [
        [0, 0, 0, .... 0], [0, 0, 0, .... 0], [0, 0, 0, .... 0]
    ]
}
```
Note that, `algo_of_each_op` indicates the tensor parallelism dimension. `0` is the default partition dimension, while `1` stands for alternative one, please refer to Sec 4.2 for more details.

### Training
To train the model with the found configs by Aceso, run the following command: 
```
## In the `Aceso/runtime` path
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --flexpipe-config CONFIG_FILE \
       --train-iters 5 \
       --eval-iters 0 \
       --lr-decay-iters 320000 \
       --vocab-file vocabs/gpt2-vocab.json \
       --merge-file vocabs/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --DDP-impl local \
       --fp16 \
       --log-path LOG_PATH
```
**Arguments:**
- `--flexpipe-config`: a config file generated by the search process.

For the other arguments, please refer to Megatron-LM's documentation.


## Troubleshooting
- **Hang at distributed training (loading fused kernel).**

    In Megatron-LM (and Aceso), some kernels are fused and the fused kernels need to be compiled at the first time. So it is OK if you are stuck at fused kernel loading when you run the distributed training for the first time. (The compilation takes ~ 10 minutes). 
    
    However, the kernel loading may get stuck if you directly copy the compiled fused kernels to other nodes. To solve this issue, please delete the fused kernel build folder (if it exists): `rm -rf runtime/megatron/fused_kernels/build`. Then the kernels will be recompiled automatically before the next time running.

- **NCCL error when training distributedly.**

    When using docker, you may need to manually set the network interface for NCCL, using the environment `NCCL_SOCKET_IFNAME`.

- **Check GPU topology when performance is not as expected.**

    Aceso always assigns adjacent GPUs to perform tensor-parallelism with the assumption that adjacent GPUs have higher bandwidth. For example, if we have 4 GPUs and want to perform 2-way tensor-parallelism and 2-way data-parallelism, Aceso will let [GPU0, GPU1] in a tensor parallelism group, and [GPU2, GPU3] in another tensor-parallelism group. 

    However, in some machines with complex GPU topology, the adjacent GPUs may not be connected via the highest bandwidth. You can manually change the GPU order using the environment `CUDA_VISIBLE_DEVICES`.

## Microsoft Open Source Code of Conduct

  This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

  Resources:
  - [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
  - [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
  - Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns