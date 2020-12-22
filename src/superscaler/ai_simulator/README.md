# AI Simulator

AI Simulator is a tool aiming to estimate the execution time for running different models (computing graphs) on different platforms (TF, PyTorch, etc.) and hardware (CPUs, GPUs, networking, etc.). It will be very useful when you would like to choose proper devices even without specific hardware or find performance bottleneck in current system.

AI Simulator provides a python package named *ai_simulator*.

## Install

You can use AI Simulator as a sub-module of SuperScaler project.

See [how to install SuperScaler project](../../../README.md).

## Run your first AI Simulator!

```python
from superscaler.ai_simulator import Simulator

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
    },
    {

    }
]
device_list = [('CPU', ["/server/hostname1/CPU/0"])]

# Init Simulator and run
sim = Simulator(node_list, device_list)
timeuse, start_time, finish_time = sim.run()

print('The total time is: %.1f' % timeuse)
```

## Run Test

You can go to SuperScaler project's root folder, and run test:

```bash
python3 -m pytest -v tests/ai_simulator
```

## Run Simulator Benchmark

See how to use [Simulator Benchmark](../../../tools/simulator_benchmark/README.md)

Simulator Benchmark can generate database coverage and simulator accuracy using 8 typical models and in 4 different environments including 1/2/4 GPU and 2 Host X 4 GPU.

## For Developers

### Project Organization

- `./simulator/`: The source code of the simulator module.
- `./READEME.md`: The README file.

You can get the design document of AI Simulator at `SuperScaler/docs/simulator_design.docx` 

### AI Simulator development

There is an interface that users will call

- In package `ai_simulator`
  - `Simulator`: input json of *node_list* and *device_list* and use `Simulator.run()` to get the final result.

### Customized Device creation

You can create you own customed device which can be merged into the simulator.

Basically you need to implement a new module for your customized device in the `simulator` package. A good example is the [`fifo_device`](https://msrasrg.visualstudio.com/SuperScaler/_git/SuperScaler?path=%2Fai_simulator%2Fsimulator%2Ffifo_device.py&version=GBdev&_a=contents) module. Basically, your customed device module should contain an inherited class from `Device` class in `device` module. And in the inherited class you need to implement the following three interfaces.

- `get_next_node(self)`: Get the first completed node
- `enqueue_node(self, node, time_now)`: Enqueue a new node into this device.
- `dequeue_node(self)`: Dequeue the first completed node from the device. Do not modify the attribute of the node, just modify info of device.

For details, please refer to `./simulator/fifo_device.py` as an example.
