import json
from pathlib import Path
from frontend.scaler_graph.IR.conversion import tensorflow as tf_adapter


def test_graph_io():
    '''test:
        import sc graph from tf pbtxt file;
        export sc graph into tf pbtxt file;
    '''
    tf_pbtxt_path = "test/tf_example/SimpleCNNRun.pbtxt"
    sc_graph = tf_adapter.import_graph_from_tf_file(tf_pbtxt_path)
    sc_graph_serialization_file = "test/tf_example/SimpleCNNRun.json"
    file = Path(sc_graph_serialization_file)
    assert(len(sc_graph.nodes) == 130)
    assert(json.loads(file.read_text()) == json.loads(sc_graph.json()))
    tf_adapter.export_to_graph_def_file(sc_graph)


def test_graph_modification():
    '''test:
        remove node and edge
    '''
    tf_pbtxt_path = "test/tf_example/MatmulRun.pbtxt"
    sc_graph = tf_adapter.import_graph_from_tf_file(tf_pbtxt_path)
    node_names = [node.name for node in sc_graph.nodes]
    for node_name in node_names:
        node = sc_graph.get_node_by_name(node_name)
        sc_graph.remove_node_and_edge(node)
    assert(len(sc_graph.nodes) == 0)
