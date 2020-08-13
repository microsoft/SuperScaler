# AI Simulator

AI Simulator is a tool aiming to estimate the execution time for running different models (computing graphs) on different platforms (TF, PyTorch, etc.) and hardware (CPUs, GPUs, networking, etc.). It will be very useful when you would like to choose proper devices even without specific hardware or find performance bottleneck in current system.

AI Simulator provides a python package named *ai_simulator*.

## Install

### Install by Dockerfile

Using AI Simulator at Docker environment is the easiest method.

```bash
# Build AI Simulator
sudo docker build -t ai-simulator .

# Run AI Simulator experiment
sudo docker run ai-simulator
```

### Install on Native Machine

Please install Python before running the AI Simulator. The recommended setting is as follows:

```text
    python: 3.6–3.7 (64 bit)
    pip: >= 19.0
    protobuf: 3.8
    pytest: 5.3.2
    setuptools: 41.0.0
    bitmath: 1.3.3.1
    humanreadable: 0.1.0
    pyYAML: 5.1.2
```

- Python

    Because TensorFlow on Windows only supports Python 3.x (64-bit), we highly recommend to install Python 3.x (64-bit). You can download and install Python from [Python Website](https://www.python.org/). You can update the pip to the latest version.

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

#### Using ai_simulator source code and installing dependencies manually

Goto the ai_simulator folder, and run:

```bash
# Install TensorFlow, Protobuf, and Pytest
python3 -m pip install -r requirements.txt

# Verify the installation
python3 -m pytest -v
```

#### Installing ai_simulator by setuptools

Using setuptools to install *superscaler* which includes not only *ai_simulator* :

Goto path *SuperScaler/* first and run:

```bash
# Install superscaler package using setuptools
python3 setup.py install

# Verify the installation
python3 -m pytest -s tests/test_integration.py
```

Currently tensorflow==1.15 may be required by plan_gen package of superscaler.

Manually install it by `python3 -m pip install tensorflow==1.15` if error reported.

## Run your first AI Simulator!

```python
from ai_simulator import Simulator

# generate example node_list and device_list
node_list = [
    {
        'dependency_ids': [],
        'device_name': '/server/hostname1/CPU/0',
        'execution_time': 1.0,
        'index': 0,
        'input_ids': [],
        'name': 'op_no_1',
        'op': 'Const',
        'successor_ids': [],
        'output_tensors': []
    },
]
device_list = [('CPU', ["/server/hostname1/CPU/0"])]

# Init Simulator and run
sim = Simulator(node_list, device_list)
timeuse, start_time, finish_time = sim.run()

print('The total time is: %.1f' % timeuse)
```

See how ai_simulator work with plan_gen at [test_integration.py](../tests/test_integration.py)

## Run Test

You can go to ai_simulator folder, and run test:

```bash
python3 -m pytest -v
```

## For Developers

### Project Organization

- `./doc/`: The detailed document of the AI Simulator.
- `./simulator/`: The source code of the simulator module.
- `./tests/`: The unit test files used in the AI Simulator.
- `./Dockerfile`: The Dockerfile used to build the docker environment for the AI Simulator.
- `./requirements.txt`: The prerequisite Python packages for the AI Simulator.
- `./READEME.md`: The README file.

### AI Simulator development

There are several interfaces that users will call

- In package `ai_simulator`
  - `PlanAdapter`: input and check the json input of *node_list*
  - `Simulator`: input *node_list* and *device_list* and use `Simulator.run()` to get the final result.

### Customized Device creation

You can create you own customed device which can be merged into the simulator.

Basically you need to implement a new module for your customized device in the `simulator` package. A good example is the [`fifo_device`](https://msrasrg.visualstudio.com/SuperScaler/_git/SuperScaler?path=%2Fai_simulator%2Fsimulator%2Ffifo_device.py&version=GBdev&_a=contents) module. Basically, your customed device module should contain an inherited class from `Device` class in `device` module. And in the inherited class you need to implement the following three interfaces.

- `get_next_node(self)`: Get the first completed node
- `enqueue_node(self, node, time_now)`: Enqueue a new node into this device.
- `dequeue_node(self)`: Dequeue the first completed node from the device. Do not modify the attribute of the node, just modify info of device.

For details, please refer to `./simulator/fifo_device.py` as an example.
