import tempfile
import os
from pathlib import Path
WORKDIR_HANDLER = tempfile.TemporaryDirectory()
os.environ["TF_DUMP_GRAPH_PREFIX"] = WORKDIR_HANDLER.name
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "4"
from frontend.scaler_graph.parallelization.parallelism import DataParallelism
from frontend.scaler_graph.parallelization.parallelizer import Parallelizer
from frontend.scaler_graph.IR.conversion import tensorflow as tf_adapter
from frontend.scaler_graph.test.tf_example import dummy_model


def test_dataparallelism():
    apply_gradient_op, loss = dummy_model.SimpleCNN()
    merged_sc_graph = tf_adapter.import_tensorflow_model(
        apply_gradient_op, loss)
    parallelizer = Parallelizer(merged_sc_graph)
    parallelizer.register_parallelism(DataParallelism(range(2)))
    parallelizer.run_parallelisms()
    inserted_allreduce_file = "test/tf_example/SimpleCNNAllreduce.json"
    file = Path(inserted_allreduce_file)
    assert (file.read_text() == parallelizer.graphs[0].json())
    assert (len(parallelizer.graphs) == 2)
