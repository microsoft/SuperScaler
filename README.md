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

### Build Docker **(TODO: setup Docker)**

**(TODO: replace with real process)**

Using SuperScaler at Docker environment is the easiest method.

```bash
# Build SuperScaler Docker
sudo docker build -t superscaler .

# Run SuperScaler Docker environment
sudo docker run superscaler
```

### Install on Native Machines

**(TODO: survey the dependencies and how to set up from start)**

Install dependencies:

``` bash
sudo apt install xxx xxx xxx
```

...

...

## Test

**(TODO: write a one-step test script to verify installation)**

One line Script to run test for SuperScaler:

```
xxx
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
