# SuperScaler

SuperScaler is an open-source distributed training platform for deep learning, which is designed to enable and accelerate the distributed and parallel training of deep neural networks (DNN). SuperScaler will support multiple deep learning frameworks. [TensorFlow](https://www.tensorflow.org/) and [NNFusion](https://github.com/microsoft/nnfusion) is supported now.

## Features of SuperScaler

- Use data parallel (model and pipeline parallel comming soon) to achieve parallel training.

- Use collective communication to accelerate distributed training.

- Solve `AllReduce` to optimal `Send` and `Receive` to improve communication performance.

- Schedule the computing plan according to different computing and communication capabilities of different devices.

- Dynamically reschedule the computing plan during executing.

- Highly-optimized computation and communication kernels.

- Broad support for different deep learning frameworks and different hardware platforms.

## Install

### Install on Native Machines

- Install dependencies (Optional, you can back to here if installation failed):

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
