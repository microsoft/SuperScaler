from pprint import pprint
from collections import namedtuple
import os
import json
import pytest

from simulator import Simulator
from simulator.tensor import Tensor
from simulator.computation_device import GPU, CPU
from simulator.network_device import NetworkSwitch


def test_simulator_naive_allreduce():
    # Naive Allreduce:
    # GPU1  -----------↘     ↗--------→    GPU1
    # GPU2  ---------→  GPU0 ---------→    GPU2
    # GPU3  -----------↗     ↘--------→    GPU3
    # Get input DAG and device info
    node_list = []
    simulator_unit_test_file_relative_path = os.path.join(
        os.path.dirname(__file__), "test_allreduce_simulator_input",
        "naive_allreduce_nodes_metadata_test.json")
    with open(simulator_unit_test_file_relative_path) as f:
        node_json_data = json.load(f)

    # Iterate the "node_list" objects, convert each json object to node
    for node_json_obj in node_json_data["node_list"]:
        node_obj = json.loads(
            json.dumps(node_json_obj),
            object_hook=lambda d: namedtuple(
                'metadata_tuple', d.keys())(*d.values()))
        tensor_list = []
        for tensor_metadata in node_obj.output_tensors:
            tensor_list.append(
                Tensor(tensor_metadata[0], tensor_metadata[1])
            )
        final_obj = node_obj._replace(output_tensors=tensor_list)
        node_list.append(final_obj)

    device_list = [
        CPU("/server/hostname1/CPU/0"),
        GPU("/server/hostname1/GPU/0"), GPU("/server/hostname1/GPU/1"),
        GPU("/server/hostname1/GPU/2"), GPU("/server/hostname1/GPU/3"),
        NetworkSwitch(
            "/switch/switch0",
            [("/server/hostname1/GPU/0", '80bps'),
             ("/server/hostname1/GPU/1", '80bps'),
             ("/server/hostname1/GPU/2", '80bps'),
             ("/server/hostname1/GPU/3", '80bps')
             ]
        )
    ]

    # Call simulator
    sim = Simulator(node_list, device_list)
    timeuse, start_time, finish_time = sim.run()

    pprint("Naive Allreduce timeuse: " + str(timeuse))
    assert timeuse == 13
    pprint(finish_time)
    assert finish_time == [(0, 6.0), (1, 6.0), (2, 6.0), (3, 7.0),
                           (4, 9.0), (5, 11.0), (6, 13.0)]
    pprint(start_time)
    assert start_time == [(0, 0.0), (1, 0.0), (2, 0.0), (3, 6.0),
                          (4, 7.0), (5, 9.0), (6, 11.0)]


def test_simulator():
    # Mock the input data (node_list)
    node_list = []
    simulator_unit_test_file_relative_path = os.path.join(
        os.path.dirname(__file__), "test_simulator_input",
        "simulator_unit_test.json")
    with open(simulator_unit_test_file_relative_path) as f:
        node_json_data = json.load(f)

    # Iterate the "node_list" objects, convert each json object to node
    for node_json_obj in node_json_data["node_list"]:
        node_obj = json.loads(
            json.dumps(node_json_obj),
            object_hook=lambda d: namedtuple(
                'metadata_tuple', d.keys())(*d.values()))
        tensor_list = []
        for tensor_metadata in node_obj.output_tensors:
            tensor_list.append(
                Tensor(tensor_metadata[0], tensor_metadata[1])
            )
        final_obj = node_obj._replace(output_tensors=tensor_list)
        node_list.append(final_obj)

    # device_list incoherence, node_list needs 3 device:
    # "/server/hostname1/CPU/0", "/server/hostname1/CPU/1" and
    # "/server/hostname1/GPU/0". However, in wrong_device_list there are only
    # 2 devices
    wrong_device_list = [
        GPU("/server/hostname1/GPU/0"),
        CPU("/server/hostname1/CPU/0"),
    ]
    with pytest.raises(TypeError):
        Simulator(node_list, wrong_device_list)
    # device_list wrong format: device_info should not be a dict
    wrong_device_info = {'GPU': GPU("/server/hostname1/GPU/0")}
    with pytest.raises(ValueError):
        Simulator(node_list, wrong_device_info)
    # device_list wrong format: device_info should be a list of ***tuple***
    wrong_device_info = [
        {'GPU': ["/server/hostname1/GPU/0"]}
    ]
    with pytest.raises(ValueError):
        Simulator(node_list, wrong_device_info)
    # Test wrong device_type
    wrong_device_list = [
        ('WRONG_DEVICE_TYPE', ["/server/hostname1/GPU/0"])
    ]
    with pytest.raises(ValueError):
        Simulator(node_list, wrong_device_list)

    device_list = [
        GPU("/server/hostname1/GPU/0"),
        CPU("/server/hostname1/CPU/0"),
        CPU("/server/hostname1/CPU/1")
    ]
    sim = {}
    timeuse = {}
    start_time = {}
    finish_time = {}
    # Call simulator with class Device
    sim['class_input'] = Simulator(node_list, device_list)
    timeuse['class_input'], start_time['class_input'], \
        finish_time['class_input'] = sim['class_input'].run()
    # To display the debug information, please use `python -m pytest -s`
    pprint(start_time['class_input'])
    pprint(finish_time['class_input'])
    pprint(timeuse['class_input'])
    assert start_time['class_input'] == [(0, 0.0), (1, 1.0), (2, 1.0)]
    assert finish_time['class_input'] == [(0, 1.0), (2, 2.0), (1, 3.0)]
    assert timeuse['class_input'] == 3

    device_tuple_list = [
        ('GPU', ["/server/hostname1/GPU/0"]),
        ('CPU', ["/server/hostname1/CPU/0"]),
        ('CPU', ["/server/hostname1/CPU/1"])
    ]

    # Call simulator with device tuple info
    sim['param_input'] = Simulator(node_list, device_tuple_list)
    timeuse['param_input'], start_time['param_input'], \
        finish_time['param_input'] = sim['param_input'].run()
    # The 'param_input' and 'class_input' denote the same devices, so
    # the results should be the same
    assert start_time['param_input'] == [(0, 0.0), (1, 1.0), (2, 1.0)]
    assert finish_time['param_input'] == [(0, 1.0), (2, 2.0), (1, 3.0)]
    assert timeuse['param_input'] == 3
