# SuperScaler

**SuperScaler** is an open-source distributed platform for deep learning training.
SuperScaler aims to provide the flexbile support of training parallelization and be extensible to new parallelisms and optimizations.
By leveraging existing deep learning frameworks like [TensorFlow](https://www.tensorflow.org/) and [NNFusion](https://github.com/microsoft/nnfusion) for local execution while supporting efficient distributed training with highly-optimized communication stacks, SuperScaler is exploring the new oppotunities of parallel deep learning training.

## Status
(alpha preview)

- Data-parallelism enabled for multi-GPU parallel training
- Support flexible communication, e.g., building `AllReduce` with primitives `Send` and `Receive` 
- TensorFlow 1.x and NNFusion supported

## Install

### Install on a Bare-metal Machine

- Install dependencies
  ```bash
  # Install tools needed
  sudo apt-get update && apt-get install build-essential wget

  # We require cmake >= 3.17 so we need to install it mannually
  sudo wget -qO- "https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

  # make sure you use python3.6 or 3.7 because
  # tensorflow 1.15 does not support python3.8 or higher.
  python3 --version

  # make sure you use tensorflow1.15 rather than tensorflow2
  pip3 install tensorflow==1.15
  python3 -c 'import tensorflow as tf; print(tf.__version__)'
  # (then '1.15.x' will be printed)
  ```

- Install from source code

  Simply use pip to build and install:

  ```bash
  git clone https://github.com/microsoft/superscaler.git
  cd superscaler
  pip3 install .
  ```

### Run with Docker

Using SuperScaler at Docker environment is the easiest method.

- Build SuperScaler Docker:

  ```bash
  sudo docker build -t superscaler -f Dockerfile.CUDA .
  ```

- Run SuperScaler Docker and run test script by default:

  ```bash
  sudo docker run --runtime=nvidia superscaler
  ```

- Or run Docker with interactive mode:

  ```bash
  sudo docker run -it --runtime=nvidia superscaler bash

  # (then, you have got into the docker‘s bash shell)
  ```

## Run your first model with SuperScaler
Here we use a TensorFlow model as an example.

- First we should create a file 'resource_pool.yaml', fill in the resource informations. You can get [a sample resource_pool.yaml](#a-sample-resource_pool.yaml) here
Then build a TensorFlow module and get the apply_gradient and loss op:
```python
# Build a TensorFlow model with returning applying_gradient_op and loss_op (ref example/tensorflow/dummy_model.py)
apply_gradient, loss = apply_gradient_op, loss_op
```
- To configure SuperScaler in code
```python
import superscaler.tensorflow as superscaler
from superscaler.scaler_graph import DataParallelism
import argparse
sc = superscaler()
strategy = DataParallelism(range(2))
deployment_setting = {"1": "localhost"}
communication_DSL = "ring"
resource_pool = "./resource_pool.yaml"
sc.init(apply_gradient_op, loss, deployment_setting, strategy,
        communication_DSL, resource_pool
```
- To run your program
```python
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()
args.steps = 10
args.interval = 5
args.print_info = True
args.print_fetches_targets = True
sc.run(args)
```
#### Appendix: A Sample resource_pool.yaml
```yaml
Server:
    hostname1:
        CPU:
            0:
                properties:
                    average_performance: "12Gibps"
                links:
                    -
                        dest: "/server/hostname1/CPU/1/"
                        type: "RDMA"
                        rate: "100Gibps"
                        propagation_latency: "20us"
                        scheduler: 'FairSharing'
            1:
                properties:
                    average_performance: "12Gibps"
                links:
                    -
                        dest: "/server/hostname1/CPU/0/"
                        type: "RDMA"
                        rate: "100Gibps"
                        propagation_latency: "20us"
                        scheduler: 'FairSharing'
        GPU:
            0:
              properties:
                  average_performance: "12Tibps"
              links:
                  -
                      dest: "/switch/switch0/"
                      type: "PCIE"
                      rate: "80bit/s"
                      propagation_latency: "2us"
                      scheduler: 'FIFO'
                  -
                      dest: "/server/hostname1/GPU/1/"
                      type: "RDMA"
                      rate: "100bit/s"
                      propagation_latency: "2us"
                      scheduler: 'FIFO'
            1:
              properties:
                  average_performance: "12Tibps"
              links:
                  -
                      dest: "/switch/switch0/"
                      type: "PCIE"
                      rate: "80bit/s"
                      propagation_latency: "2us"
                      scheduler: 'FIFO'
                  -
                      dest: "/server/hostname1/GPU/0/"
                      type: "RDMA"
                      rate: "100bit/s"
                      propagation_latency: "2us"
                      scheduler: 'FIFO'
 
Switch:
    switch0:
        links:
            -
                dest: "/server/hostname1/GPU/0/"
                type: "PCIE"
                rate: "80bit/s"
                propagation_latency: "2us"
                scheduler: 'FIFO'
            -
                dest: "/server/hostname1/GPU/1/"
                type: "PCIE"
                rate: "80bit/s"
                propagation_latency: "2us"
                scheduler: 'FIFO'
```

- ## Microsoft Open Source Code of Conduct

  This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

  Resources:
  - [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
  - [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
  - Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns
