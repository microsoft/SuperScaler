import os
import json
import pytest
import tensorflow as tf
from google.protobuf import text_format
from adapter.tf_parser import TFNodeAttrParser, TFParser, ParserError

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression):
        self.expression = expression

def test_tf_attr_parser():

    node_wrong_bool_path = "tests/data/tf_parser_testbench/node_wrong_bool_value"
    node_wrong_type_path = "tests/data/tf_parser_testbench/node_wrong_type"
    node_wrong_str_path = "tests/data/tf_parser_testbench/node_wrong_str"
    node_wrong_float_path = "tests/data/tf_parser_testbench/node_wrong_float"
    node_wrong_shape_path = "tests/data/tf_parser_testbench/node_wrong_shape"
    node_wrong_tensor_path = "tests/data/tf_parser_testbench/node_wrong_tensor"
    node_standard_path = "tests/data/tf_parser_testbench/node_standard"

    def test_node(filename):
        def __load_protobuf_from_file(filename):
            graph_def = None
            with open(filename, 'r') as f:
                file_content = f.read()
                try:
                    graph_def = text_format.Parse(file_content, tf.compat.v1.GraphDef())
                    return graph_def
                except text_format.ParseError as e:
                    raise InputError("Cannot parse file %s: %s."
                                    % (filename, str(e)))
            return graph_def

        graph = __load_protobuf_from_file(filename)
        node  = graph.node[0]
        parser = TFNodeAttrParser()
        attr = parser.parse_node(node)
        attr_str = parser.parse_node(node)
        return attr

    with pytest.raises(ParserError):
        "None input test"
        parser = TFNodeAttrParser()
        parser.parse_node(None)
    with pytest.raises(InputError):
        "Node with non-bool value"
        test_node(node_wrong_bool_path)
    with pytest.raises(InputError):
        "Node with wrong type"
        test_node(node_wrong_type_path)
    with pytest.raises(InputError):
        "Node with non-str value"
        test_node(node_wrong_str_path)
    with pytest.raises(InputError):
        "Node with non-float value"
        test_node(node_wrong_float_path)
    with pytest.raises(InputError):
        "Node with empty shapes"
        test_node(node_wrong_shape_path)
    with pytest.raises(InputError):
        "Node with error tensor"
        test_node(node_wrong_tensor_path)


    "standard test, pass pytest"
    attr = test_node(node_standard_path)
    assert(attr['T'] == 1)
    assert(attr['_output_shapes'] == [[5,5,64,64]])
    assert(attr['num_devices'] == '2')
    assert(attr['reduction'] == 'sum')
    assert(attr['tensor_name'] == "For_gradients/conv1/conv2d/Conv2D_grad/tuple/control_dependency_1")


def test_TFParser():

    def get_device(device_count):
        return ["device_%d"%(i) for i in range(device_count)]
    
    def get_graph_paths(path, device_count):
        path = os.path.join(os.path.dirname(__file__), path)
        graph_paths = []
        for i in range(device_count):
            sub_path = os.path.join(path, "run_" + str(i) + ".pbtxt")
            graph_paths.append(sub_path)
        return graph_paths

    with pytest.raises(Exception):
        "Test benchmark using 2 devices and 3 graphs, raise Exception"
        parser = TFParser()
        devices = get_device(2)
        graph_paths = get_graph_paths("data/DataParallelismPlan2GPUsIn2Hosts", 3)
        parser = TFParser()
        plans = parser.parse_graphs(graph_paths, devices)

    with pytest.raises(Exception):
        "Test benchmark using 3 devices and 2 graphs, raise Exception"
        parser = TFParser()
        devices = get_device(3)
        graph_paths = get_graph_paths("data/DataParallelismPlan2GPUsIn2Hosts", 2)
        parser = TFParser()
        plans = parser.parse_graphs(graph_paths, devices)

    "Test benchmark using 2 devices and 2 graphs, pass pytest"
    device_count = 2
    parser = TFParser()
    devices = get_device(device_count)
    graph_paths = get_graph_paths("data/DataParallelismPlan2GPUsIn2Hosts", device_count)
    parser = TFParser()
    plans = parser.parse_graphs(graph_paths, devices)


    ref_path = os.path.join( os.path.join(os.path.dirname(__file__),"data/DataParallelismPlan2GPUsIn2Hosts"), "Nodes.json")
    ref_plans = json.load(open(ref_path, "r"))
    assert(plans == ref_plans)

