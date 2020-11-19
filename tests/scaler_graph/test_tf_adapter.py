import os
import tempfile
import subprocess
import google.protobuf.text_format
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
from superscaler.scaler_graph.IR.conversion import tf_adapter  # noqa: E402
from tf_example import dummy_model  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.python import pywrap_tensorflow  # noqa: E402


def is_cuda_available():
    """Check NVIDIA with nvidia-smi command
    Returning code 0 if no error, it means NVIDIA is installed
    Other codes mean not installed
    """
    code = os.system('nvidia-smi')
    if code == 0:
        cmd = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
        count = subprocess.check_output(cmd, shell=True)
        return int(count) > 0
    else:
        return False


def test_tf_adapter():
    tf.reset_default_graph()
    # import sc graph from tf model;
    apply_gradient_op, loss = dummy_model.SimpleCNN()
    WORKDIR_HANDLER = tempfile.TemporaryDirectory()
    merged_sc_graph = tf_adapter.import_tensorflow_model(
        apply_gradient_op, loss, WORKDIR_HANDLER.name)
    assert (merged_sc_graph is not None)
    # export sc graph to tf model;
    if is_cuda_available():
        tf_pbtxt_path = os.path.join(os.path.dirname(__file__),
                                     "data", "SimpleCNN.pbtxt")
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
        assert (len(diff) == 0)
