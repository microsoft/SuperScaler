from pprint import pprint
from collections import namedtuple
import os
import copy
import json
import pytest

from simulator import Simulator
from simulator.tensor import Tensor
from simulator.computation_device import GPU, CPU


def test_simulator():
    # Mock the input data (node_list)
    node_tuple_list = []
    node_dict_list = []
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
        final_tuple_obj = node_obj._replace(output_tensors=tensor_list)
        # Init node_tuple_list
        node_tuple_list.append(final_tuple_obj)
        # Init node_dict_list
        final_dict_obj = copy.deepcopy(node_json_obj)
        final_dict_obj['output_tensors'] = tensor_list
        node_dict_list.append(final_dict_obj)

    # device_list incoherence, node_list needs 3 device:
    # "/server/hostname1/CPU/0", "/server/hostname1/CPU/1" and
    # "/server/hostname1/GPU/0". However, in wrong_device_list there are only
    # 2 devices
    wrong_device_list = [
        GPU("/server/hostname1/GPU/0"),
        CPU("/server/hostname1/CPU/0"),
    ]
    with pytest.raises(TypeError):
        Simulator(node_tuple_list, wrong_device_list)
    # device_list wrong format: device_info should not be a dict
    wrong_device_info = {'GPU': GPU("/server/hostname1/GPU/0")}
    with pytest.raises(ValueError):
        Simulator(node_tuple_list, wrong_device_info)
    # device_list wrong format: device_info should be a list of ***tuple***
    wrong_device_info = [
        {'GPU': ["/server/hostname1/GPU/0"]}
    ]
    with pytest.raises(ValueError):
        Simulator(node_tuple_list, wrong_device_info)
    # Test wrong device_type
    wrong_device_list = [
        ('WRONG_DEVICE_TYPE', ["/server/hostname1/GPU/0"])
    ]
    with pytest.raises(ValueError):
        Simulator(node_tuple_list, wrong_device_list)

    device_list = [
        GPU("/server/hostname1/GPU/0"),
        CPU("/server/hostname1/CPU/0"),
        CPU("/server/hostname1/CPU/1")
    ]

    # Test wrong nodemetadata_list
    with pytest.raises(ValueError):
        # nodemetadata_list should be a list
        Simulator({}, device_list)
    with pytest.raises(ValueError):
        # nodemetadata_list should be a list of namedtuple/dict
        Simulator([[]], device_list)
    with pytest.raises(KeyError):
        # elements in nodemetadata_list should have essential attributes
        Simulator([node_dict_list[0], {}], device_list)
    with pytest.raises(ValueError):
        # The second element is not a dict/tuple
        Simulator([node_dict_list[0], []], device_list)
    # A correct example, this should not raise an error
    Simulator([node_dict_list[0], *node_tuple_list[1:]], device_list)

    sim = {}
    timeuse = {}
    start_time = {}
    finish_time = {}
    # Call simulator with list of namedtuple metadata and class Device
    sim['class_input'] = Simulator(node_tuple_list, device_list)
    timeuse['class_input'], start_time['class_input'], \
        finish_time['class_input'] = sim['class_input'].run()
    # To display the debug information, please use `python -m pytest -s`
    pprint(start_time['class_input'])
    pprint(finish_time['class_input'])
    pprint(timeuse['class_input'])
    assert start_time['class_input'] == [(0, 0.0), (1, 1.0), (2, 1.0)]
    assert finish_time['class_input'] == [(0, 1.0), (2, 2.0), (1, 3.0)]
    assert timeuse['class_input'] == 3

    # Call simulator with list of dict metadata and class Device
    sim['dict_class_input'] = Simulator(node_dict_list, device_list)
    timeuse['dict_class_input'], start_time['dict_class_input'], \
        finish_time['dict_class_input'] = sim['dict_class_input'].run()
    assert start_time['dict_class_input'] == [(0, 0.0), (1, 1.0), (2, 1.0)]
    assert finish_time['dict_class_input'] == [(0, 1.0), (2, 2.0), (1, 3.0)]
    assert timeuse['dict_class_input'] == 3

    device_tuple_list = [
        ('GPU', ["/server/hostname1/GPU/0"]),
        ('CPU', ["/server/hostname1/CPU/0"]),
        ('CPU', ["/server/hostname1/CPU/1"])
    ]

    # Call simulator with device tuple info
    sim['param_input'] = Simulator(node_tuple_list, device_tuple_list)
    timeuse['param_input'], start_time['param_input'], \
        finish_time['param_input'] = sim['param_input'].run()
    # The 'param_input' and 'class_input' denote the same devices, so
    # the results should be the same
    assert start_time['param_input'] == [(0, 0.0), (1, 1.0), (2, 1.0)]
    assert finish_time['param_input'] == [(0, 1.0), (2, 2.0), (1, 3.0)]
    assert timeuse['param_input'] == 3
