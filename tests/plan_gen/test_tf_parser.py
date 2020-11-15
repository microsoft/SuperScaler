import os
import json
import pytest
import tensorflow as tf
from google.protobuf import text_format
from superscaler.plan_gen.plan.parser.tf_parser import TFNodeAttrParser,\
    TFParser, ParserError

TEST_DB_FILE = os.path.join(
    os.path.dirname(__file__), 'data/tf_parser_testbench/db_test.json')


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression):
        self.expression = expression


def test_tf_attr_parser():

    def test_node(filename):
        def __load_protobuf_from_file(filename):
            graph_def = None
            with open(filename, 'r') as f:
                file_content = f.read()
                try:
                    graph_def = text_format.Parse(file_content,
                                                  tf.compat.v1.GraphDef())
                    return graph_def
                except text_format.ParseError as e:
                    raise InputError("Cannot parse file %s: %s." %
                                     (filename, str(e)))
            return graph_def

        graph = __load_protobuf_from_file(filename)
        node = graph.node[0]
        parser = TFNodeAttrParser()
        attr = parser.parse_node(node)
        return attr

    with pytest.raises(ParserError):
        "None input test"
        parser = TFNodeAttrParser()
        parser.parse_node(None)
    with pytest.raises(InputError):
        "Node with non-bool value"
        path = os.path.join(
            os.path.dirname(__file__),
            "data/tf_parser_testbench/node_wrong_bool_value")
        test_node(path)
    with pytest.raises(InputError):
        "Node with wrong type"
        path = os.path.join(
            os.path.dirname(__file__),
            "data/tf_parser_testbench/node_wrong_type")
        test_node(path)
    with pytest.raises(InputError):
        "Node with non-str value"
        path = os.path.join(
            os.path.dirname(__file__),
            "data/tf_parser_testbench/node_wrong_str")
        test_node(path)
    with pytest.raises(InputError):
        "Node with non-float value"
        path = os.path.join(
            os.path.dirname(__file__),
            "data/tf_parser_testbench/node_wrong_float")
        test_node(path)
    with pytest.raises(InputError):
        "Node with empty shapes"
        path = os.path.join(
            os.path.dirname(__file__),
            "data/tf_parser_testbench/node_wrong_shape")
        test_node(path)
    with pytest.raises(InputError):
        "Node with error tensor"
        path = os.path.join(
            os.path.dirname(__file__),
            "data/tf_parser_testbench/node_wrong_tensor")
        test_node(path)

    standard_path = os.path.join(
        os.path.dirname(__file__),
        "data/tf_parser_testbench/node_standard")
    "standard test, pass pytest"
    attr = test_node(standard_path)
    assert(attr['T'] == 1)
    assert(attr['_output_shapes'] == [[5, 5, 64, 64]])
    assert(attr['num_devices'] == '2')
    assert(attr['reduction'] == 'sum')
    assert(attr['tensor_name'] ==
           "For_gradients/conv1/conv2d/Conv2D_grad/tuple/control_dependency_1")


def test_TFParser():

    def get_device(device_count):
        return ["device_%d" % (i) for i in range(device_count)]

    def get_graph_paths(path, device_count):
        path = os.path.join(os.path.dirname(__file__), path)
        graph_paths = []
        for i in range(device_count):
            sub_path = os.path.join(path, "run_" + str(i) + ".pbtxt")
            graph_paths.append(sub_path)
        return graph_paths

    with pytest.raises(Exception):
        "Test benchmark using 2 devices and 3 graphs, raise Exception"
        parser = TFParser(db_file_path=TEST_DB_FILE)
        devices = get_device(2)
        graph_paths = get_graph_paths(
            "data/DataParallelismPlan2GPUsIn2Hosts", 3)
        parser = TFParser(db_file_path=TEST_DB_FILE)
        nodelist = parser.parse_graphs(graph_paths, devices)

    with pytest.raises(Exception):
        "Test benchmark using 3 devices and 2 graphs, raise Exception"
        parser = TFParser(db_file_path=TEST_DB_FILE)
        devices = get_device(3)
        graph_paths = get_graph_paths(
            "data/DataParallelismPlan2GPUsIn2Hosts", 2)
        parser = TFParser(db_file_path=TEST_DB_FILE)
        nodelist = parser.parse_graphs(graph_paths, devices)

    with pytest.raises(Exception):
        "Test benchmark from file but using load_from_memory, raise Exception"
        parser = TFParser(db_file_path=TEST_DB_FILE)
        devices = get_device(2)
        graph_paths = get_graph_paths(
            "data/DataParallelismPlan2GPUsIn2Hosts", 2)
        parser = TFParser(db_file_path=TEST_DB_FILE)
        nodelist = parser.parse_graphs(graph_paths,
                                       devices,
                                       load_from_memory=True)

    with pytest.raises(Exception):
        "Test benchmark from load_from_memory but using file, raise Exception"
        parser = TFParser(db_file_path=TEST_DB_FILE)
        devices = get_device(2)
        graph_paths = get_graph_paths(
            "data/DataParallelismPlan2GPUsIn2Hosts", 2)
        graphs = []
        for graph_path in graph_paths:
            with open(graph_path, 'r') as f:
                graphs.append(f.read())
        parser = TFParser(db_file_path=TEST_DB_FILE)
        nodelist = parser.parse_graphs(graphs,
                                       devices,
                                       load_from_memory=False)

    "Test benchmark using 2 devices and 2 graphs from file, pass pytest"
    device_count = 2
    parser = TFParser(db_file_path=TEST_DB_FILE)
    devices = get_device(device_count)
    graph_paths = get_graph_paths(
        "data/DataParallelismPlan2GPUsIn2Hosts", device_count)
    parser = TFParser(db_file_path=TEST_DB_FILE)
    nodelist = parser.parse_graphs(graph_paths, devices)
    ref_path = os.path.join(
        os.path.dirname(__file__),
        "data/DataParallelismPlan2GPUsIn2Hosts/Nodes.json")

    ref_nodelist = json.load(open(ref_path, "r"))
    assert(nodelist == ref_nodelist)

    "Test benchmark using 2 devices and 2 graphs from memory, pass pytest"
    device_count = 2
    parser = TFParser(db_file_path=TEST_DB_FILE)
    devices = get_device(device_count)
    graph_paths = get_graph_paths(
        "data/DataParallelismPlan2GPUsIn2Hosts", device_count)
    graphs = []
    for graph_path in graph_paths:
        with open(graph_path, 'r') as f:
            graphs.append(f.read())
    parser = TFParser(db_file_path=TEST_DB_FILE)
    nodelist = parser.parse_graphs(graphs, devices, load_from_memory=True)
    ref_path = os.path.join(
        os.path.dirname(__file__),
        "data/DataParallelismPlan2GPUsIn2Hosts/Nodes.json")

    ref_nodelist = json.load(open(ref_path, "r"))
    assert(nodelist == ref_nodelist)
