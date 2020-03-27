import json
import os
from pprint import pprint
from collections import namedtuple
from simulator import Simulator

def test_simulator():
    # Mock the input data (node_list)
    node_list = []
    simulator_unit_test_file_relative_path = os.path.join(
        os.path.dirname(__file__), "test_simulator_input","simulator_unit_test.json")
    simulator_unit_test_file_absolute_path = os.path.abspath(os.path.expanduser(simulator_unit_test_file_relative_path))
    with open(simulator_unit_test_file_absolute_path) as f:
        test_input_data = json.load(f)
    
    # Iterate the json object, convert each json object to node object
    for node_json_item in test_input_data["node_list"]:
        node_obj = json.loads(json.dumps(node_json_item), object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        node_list.append(node_obj)
    
    # Call simulator
    sim = Simulator(node_list)
    timeuse, execution_list = sim.run()

    # To display the debug information, please use `python -m pytest -s`
    pprint(execution_list)
    assert execution_list == [[0, 0.0], [1, 1.0], [2, 1.0]]
    pprint(timeuse)
    assert timeuse == 3