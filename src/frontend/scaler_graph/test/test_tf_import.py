import tempfile
import os
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
    sc_graph_init, sc_graph_run = tf_adapter.import_tensorflow_model(
        apply_gradient_op, loss)
    assert ((sc_graph_init is not None) and (sc_graph_run is not None))
