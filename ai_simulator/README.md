# AI Simulator

## Project Organization

* `./doc/`: The detailed document of the AI Simulator.
* `./examples/`: The tensorflow graph examples, including 4 CNN models and 4 RNN models.
* `./adapter/`: The source code of the adapter module.
* `./profiler/`: The source code of the profiler module.
* `./simulator/`: The source code of the simulator module.
* `./tests/`: The unit test files used in the AI Simulator.
* `./Dockerfile`: The Dockerfile used to build the docker environment for the AI Simulator.
* `./requirements.txt`: The prerequisite Python packages for the AI Simulator.
* `./run.py`: The command line tool for the AI Simulator.
* `./READEME.md`: The README file.

## Installation

### Install by Dockerfile (Recommended)

Using AI Simulator at Docker environment is the easiest and recommended method.
```bash
# Build AI Simulator
sudo docker build -t ai-simulator .

# Run AI Simulator experiment
sudo docker run ai-simulator
```

### Install on Native Machine

Please install Python and TensorFlow berfore running the AI Simultor. The recommended setting is as follows:

```text
    Python: 3.5–3.7 (64 bit)
    Pip: >= 19.0
    TensorFlow: 1.15
    Protobuf: 3.8
    Pytest: 5.3.2
```
- Python

    Because TensorFlow on Windows only supports Python 3.x (64-bit), we highly recommend to insatll Python 3.x (64-bit). You can download and install Python from [Python Website](https://www.python.org/). You can update the pip to latest version.

    ```bash
    # Install python on Ubuntu
    apt-get -y install python3-dev python3-pip

    # Install pip3
    python3 -m pip install --upgrade pip

    # Check your version
    python3 --version
    pip3 --version
    ```

- Python Virtual Environment (Optional, Recommended)

    Python virtual environments are used to isolate package installation from the system.

    ```bash
    # Install virtualenv
    pip3 install -U pip virtualenv
    virtualenv --version

    # Create a new virtual environment under .\venv directory
    virtualenv --system-site-packages -p python3 ./venv

    # Activate the environment
    # For Windows
    .\venv\Scripts\activate
    # For Ubuntu
    source ./venv/bin/activate

    # Deactivate the environment
    deactivate
    ```

- TensorFlow, Protobuf, and Pytest
    Goto the the ai_simulator folder, and run:

    ```bash
    # Install TensorFlow, Protobuf, and Pytest
    python3 -m pip install -r requirements.txt

    # Verify the installation
    python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    python3 -m pytest -v
    ```

## Run Test

You can go to ai_simulator folder, and run test:

```bash
python3 -m pytest -v
```

## Run AI Simulator

You can go to ai_simulator folder, and run the command line tool for the AI Simulator:
```bash
python3 run.py -g <PATH_TO_MODEL_GRAPH>.
```
The `<PATH_TO_MODEL_GRAPH>` can be either a file path or a directory path. If `<PATH_TO_MODEL_GRAPH>` is a file path, it will simulate the single model that the path points to. If `<PATH_TO_MODEL_GRAPH>` is a directory path, it will simulate all the models under the directory recusively. By default, the `<PATH_TO_MODEL_GRAPH>` is set to the absolte path to the examples folder. A log file will be generated as well for detailed information of the simulation.

For example, to run the simulator for the CNN_alexnet_cifar10 model in the `./examples/`, 
```bash
python3 run.py -g ./examples/CNN_alexnet_cifar10.pbtxt
```

For example, to run the simulator for all the models in the `./examples/`, 
```bash
python3 run.py -g ./examples/
```

For details of the usage,
```bash
python3 run.py -h
```

## For developers

### AI Simulator development
There are several interfaces from different modules that the ai simulator command line tool needs to call.

- In package `adapter`
  - In module `tf_adapter`
    - `TFAdapter.parse_protobuf_graph(graph_def)`: Parse tensorflow graph.
- In package `profiler`
  - In module `Profiler`
    - `Profiler.get_node_execution_time(node)`: Profile single node in the graph.
- In package `simulator`
  - In module `simulator`
    - `Simulator.__init__(model_graph)`: Initialize simulator for given model graph.
    - `Simulator.run()`: Run the simulator.

### Customized Device creation
You can create you own customed device which can be merged into the simulator.

Basically you need to implement a new module for your customized device in the `simulator` package. A good example is the [`fifo_device`](https://msrasrg.visualstudio.com/SuperScaler/_git/SuperScaler?path=%2Fai_simulator%2Fsimulator%2Ffifo_device.py&version=GBdev&_a=contents) module. Basically, your customed device module should contain an inherited class from `Device` class in `device` module. And in the inherited class you need to implement the following three interfaces.
- `get_next_node(self)`: Get the first completed node
- `enqueue_node(self, node, time_now)`: Enqueue a new node into this device.
- `dequeue_node(self)`: Dequeue the first completed node from the device. Do not modify the attribute of the node, just modify info of device.

For details, please refer to `./simulator/fifo_device.py` as an example.
