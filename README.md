# SuperScaler

**SuperScaler** is an open-source distributed platform for deep learning training.
SuperScaler aims to provide transparent distributed training support for different platforms with highly adaption to new emerging parallelism algorithms and optimizations.
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

**(TODO: write a minimized program case as a quickstart of SuperScaler project)**

Create a `xxx.xx` file and type in this program:

```
xxx
xxxx
xxxxx
...
```

Run this program:

```
xxx
```

## Resources

[SuperScaler Examples](https://github.com/microsoft/)

[SuperScaler User Manuals](https://github.com/microsoft/)

[SuperScaler Internals](https://github.com/microsoft/)


