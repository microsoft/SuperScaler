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
  wget -qO- "https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz" | sudo tar --strip-components=1 -xz -C /usr/local

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

- Or run Docker with interactive mode:

  ```bash
  sudo docker run -it --runtime=nvidia superscaler bash

  # (then, you have got into the docker‘s bash shell)
  ```

## Run your first model with SuperScaler

Here we use a TensorFlow model as an example.

- First we should create a file 'resource_pool.yaml', and fill in the resource information. You can get [a sample resource_pool.yaml](./example/sample_resource_pool.yaml) here.

- Then build a tensorflow model and get the train_op and loss_op. You can get [a sample tensorflow model](./example/tensorflow/MLP_model.py) here.

- Finally set up and run the superscaler with this tensorflow model like this ↓

  ```python
  import superscaler.tensorflow as superscaler
  from superscaler.scaler_graph import DataParallelism
  import argparse

  # Here should be a tensorflow model. You can replace it with your own.
  def tensorflow_model():
      ...
      ...

      # return the train op and loss op, for superscaler to run this model
      return train_op, loss_op

  sc = superscaler()

  # To configure SuperScaler

  train_op, loss_op = tensorflow_model()
  strategy = DataParallelism(range(2))
  deployment_setting = {"1": "localhost"}
  communication_DSL = "ring"
  resource_pool = "resource_pool.yaml"

  sc.init(train_op, loss_op, deployment_setting, strategy,
          communication_DSL, resource_pool)

  # To run your program

  parser = argparse.ArgumentParser()
  args, _ = parser.parse_known_args()

  args.steps = 10
  args.interval = 5
  args.print_info = True
  args.print_fetches_targets = True

  sc.run(args)
  ```

- ## Microsoft Open Source Code of Conduct

  This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

  Resources:
  - [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
  - [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
  - Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns
