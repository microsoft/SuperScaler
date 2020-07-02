import os
import json
from plan.node_list import Node
from plan.node_list import NodeList


def test_node_list():
    # Test None Input
    Node_list_test = NodeList(None)
    assert(Node_list_test.to_json() == [])

    # Load input node_list and test to_json function
    path_input = os.path.join(os.path.dirname(__file__),
                              "data/test_input_nodes.json")
    nodes = json.load(open(path_input, 'r'))

    Node_list = NodeList(nodes)
    assert(Node_list.to_json() == nodes)
    assert(len(Node_list) == 2)

    # Test insert function
    node_example = {
        "device": "device_3",
        "name": "test",
        "op": "Allreduce",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": "DT_FLOAT",
        "input": []
    }
    Node_example = Node(node_example)

    Node_list.insert(0, Node_example)

    # Test insert index and get_node function
    assert(Node_list.index(Node_example) == 0)
    assert(Node_list.get_node(0) == Node_example)
    assert(Node_list.get_node(4) is None)

    # Test remove function
    Node_list.remove(Node_example)
    assert(Node_list.index(Node_example) is None)

    # Test append function
    Node_list.append(Node_example)
    assert(Node_list.index(Node_example) == 2)


def test_node():

    # Test None Input
    Node_test = Node(None)
    assert(Node_test.to_json() == {})

    # Test wrong Input
    Node_test = Node({"wrong_key": "wrong_value"})
    assert(Node_test.to_json() == {})

    # Test sample input and to_json function
    node_example = {
        "device": "device_0",
        "name": "test",
        "op": "Allreduce",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": "DT_FLOAT",
        "input": []
    }

    Node_example = Node(node_example)
    assert(Node_example.to_json() == node_example)

    # Test __eq__function
    node_eq = {
        "device": "device_0",
        "name": "test",
        "op": "Allreduce",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": "DT_FLOAT",
        "input": []
    }

    Node_eq = Node(node_eq)
    assert(Node_example == Node_eq)
