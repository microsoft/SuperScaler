import tempfile
import os
from pathlib import Path
import json
import google.protobuf.text_format
WORKDIR_HANDLER = tempfile.TemporaryDirectory()
os.environ["TF_DUMP_GRAPH_PREFIX"] = WORKDIR_HANDLER.name
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "4"
from frontend.scaler_graph.IR.conversion import tf_adapter
from frontend.scaler_graph.test.tf_example import dummy_model
from frontend.scaler_graph.util.log import logger
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def is_cuda_available():
    """Check NVIDIA with nvidia-smi command
    Returning code 0 if no error, it means NVIDIA is installed
    Other codes mean not installed
    """
    code = os.system('nvidia-smi')
    if code != 0:
        logger("scaler_graph_test").warning(
            "It is not a CUDA-enable environment.")
    return code == 0


def test_tf_import():
    # import sc graph from tf model;
    apply_gradient_op, loss = dummy_model.SimpleCNN()
    merged_sc_graph = tf_adapter.import_tensorflow_model(
        apply_gradient_op, loss)
    assert (merged_sc_graph is not None)
    # test: get tf runtime config
    config = tf_adapter.get_tf_runtime_config(merged_sc_graph)
    config_file = "test/tf_example/SimpleCNN_model_desc.json"
    file = Path(config_file)
    assert (file.read_text() == json.dumps(config, indent=4, sort_keys=True))
    # export sc graph to tf model;
    if is_cuda_available():
        tf_pbtxt_path = "test/tf_example/SimpleCNN.pbtxt"
        test_tf_graph_def = tf.GraphDef()
        google.protobuf.text_format.Parse(
            open(tf_pbtxt_path).read(), test_tf_graph_def)
        curr_tf_graph_def = tf.GraphDef()
        google.protobuf.text_format.Parse(
            tf_adapter.export_graph_to_tf_file(merged_sc_graph),
            curr_tf_graph_def)
        diff = pywrap_tensorflow.EqualGraphDefWrapper(
            test_tf_graph_def.SerializeToString(),  # expected
            curr_tf_graph_def.SerializeToString())  # actual
        print(diff)
        assert (len(diff) == 0)
