import tempfile
import os
from pathlib import Path
import json
WORKDIR_HANDLER = tempfile.TemporaryDirectory()
os.environ["TF_DUMP_GRAPH_PREFIX"] = WORKDIR_HANDLER.name
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "4"
from frontend.scaler_graph.IR.conversion import tensorflow as tf_adapter
from frontend.scaler_graph.test.tf_example import dummy_model


def test_tf_import():
    '''test:
        import sc graph from tf model;
    '''
    apply_gradient_op, loss = dummy_model.SimpleCNN()
    merged_sc_graph = tf_adapter.import_tensorflow_model(
        apply_gradient_op, loss)
    assert (merged_sc_graph is not None)


def test_tf_runtime_config():
    # test: tf merged import
    tf_init_pbtxt_path = "test/tf_example/SimpleCNNInit.pbtxt"
    tf_run_pbtxt_path = "test/tf_example/SimpleCNNRun.pbtxt"
    merged_sc_graph = tf_adapter.import_graph_from_tf_file(
        tf_init_pbtxt_path, tf_run_pbtxt_path)
    sc_graph_serialization_file = "test/tf_example/SimpleCNN.json"
    file = Path(sc_graph_serialization_file)
    assert (len(merged_sc_graph.nodes) == 41 + 130 - 10)
    assert (json.loads(file.read_text()) == json.loads(merged_sc_graph.json()))
    # test: get tf runtime config
    config = tf_adapter.get_tf_runtime_config(merged_sc_graph)
    config_file = "test/tf_example/SimpleCNN_model_desc.json"
    file = Path(config_file)
    assert (file.read_text() == json.dumps(config, indent=4, sort_keys=True))
