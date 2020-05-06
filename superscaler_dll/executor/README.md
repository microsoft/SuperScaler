# Executor lib

This is a library providing asynchronous interface for message passing tasks.

## Overview

This library aims to provide an high-efficient asynchronous library for message transmission between memory, GPU and different machines. 

This library have three mainly basic classes: Executor, Worker and Task. 

* Executor manages a set of workers and offers interface adding tasks. 
* Worker is a working thread having its own task queue. Executor manage the tasks in workers’ task queues. Worker will take tasks from its task queues when finished all the task they took last time
* Task manages an asynchronous task and its callback function will be called when finished.

In one word, executor will create and schedule a set of workers execute tasks asynchronously, then will call tasks’ callback function when finished.

> Caution: tasks may not execute in order

## Requirement

These are basic requirement to build and test this library.

* CMake
* g++ >= 4.8.1 (c++11 needed)
* Google Test
* gtest-parallel
* Pyhton (needed by gtest-parallel)

## Build

### Build by Dockerfile

Using and testing async_common in Docker environment is the easiest and recommended method.

First, please install the docker on your system according to [Docker Website](https://docs.docker.com/install/).

```bash
# Uninstall old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Set up the repository
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# Install the docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# Verify docker
sudo docker run hello-world
```

Then you can build docker image and run the container under `async_common_lib` folder.

```bash
# Go to library folder
cd this_library_directory

# Build the docker image
sudo docker build -t async_common_lib .
```

### Build on native machine

Please install `cmake`, `g++>=4.8.1` and `googletest` before build the library.

Install g++ and cmake

```bash
sudo apt-get update -y
sudo apt-get install -y build-essential cmake python
```

Install googletest. Googletest will be installed by cmake in current project folder, or you can install googletest by executing following command if you want to use it for other projects.

```bash
# The googletest version, 1.10.x is the lastest release
export BRANCH_OR_TAG=v1.10.x

git clone --depth=1 -b $BRANCH_OR_TAG -q https://github.com/google/googletest.git googletest
mkdir -p googletest/build
cd googletest/build
cmake .. && make
sudo make install
cd ../.. && rm -rf googletest
```

Use `cmake` to build this project. Supported options: 

* `DO_TEST`: build the unit test
* `REPEAT_TIMES`: Test times for gtest_parallel, default value is 100. Unit test will be executed `REPEAT_TIMES` on each worker
* `TEST_WORKERS`: Number of workers execute unit test parallel for gtest_parallel. Default value is the number of processors or 1 if cannot get the number of processors.

```bash
mkdir build
cd build
cmake ..
# or cmake .. -DDO_TEST=true for test
```

## Usage

### Develop a task

To develop your own task, you should drive your own Task class by the way of public inheritance and override its execute function.

Your task object will be called asynchronously and callback function will be called when finished.

### Run a task 

To run a task, call add_task, the member function of Executor class. Then the task will be added to the executor waiting for execution. The callback function will be called when finished.


### Executor

Executor will execute tasks asynchronously. 

PollExecutor is driven from executor that hold a thread poll, schedule a set of workers and execute tasks. PollExecutor is highly recommended for normal cases.

## Test

### Test by Docker

Firstly, you need to build the container following the guide in [Build](##Build) section. The unit test will be executed when build the container. If you want to execute once more, you can run the container and run the test again in the container.

```bash
sudo docker run -it --rm async_common_lib

# In the container
cd build
# Run the unit test parallel
make gtest_parallel

#Or you can run unit test only one time by
make g_test

# exit from container, the container will be destoried automaticly as the option `--rm` in docker run command.
exit
```

### Test on native machine

To build the test file, the `cmake` command should be extended with `-DDO_TEST` option

Following commands build and run unit test for this library.

```bash
cd this_library_directory
mkdir build
cd build
cmake .. -DDO_TEST=true
make gtest_parallel
```

### Several unit test will be run. Including:

* Allreduce test: Using Send and Receive to build a simple all reduce task

* Poll executor test: Test the poll executor including add simple task and recursively add task

* Task class test, execute a task and wait until it finished.

* Worker test


## TODO

* CPU version send/receive

* RDMA send/receive

* CUDA IPC send/receive




