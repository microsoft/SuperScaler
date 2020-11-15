import json
import os
from pathlib import Path
from superscaler.scaler_graph.parallelization.parallelism import \
     DataParallelism
from superscaler.scaler_graph.parallelization.parallelizer import Parallelizer
from superscaler.scaler_graph.IR.conversion import tf_adapter


def test_dataparallelism():
    # test: tf merged import
    tf_init_pbtxt_path = os.path.join(os.path.dirname(__file__),
                                      "data", "SimpleCNNInit.pbtxt")
    tf_run_pbtxt_path = os.path.join(os.path.dirname(__file__),
                                     "data", "SimpleCNNRun.pbtxt")
    merged_sc_graph = tf_adapter.import_graph_from_tf_file(
        tf_init_pbtxt_path, tf_run_pbtxt_path)
    sc_graph_serialization_file = \
        os.path.join(os.path.dirname(__file__),
                     "data", "SimpleCNN.json")
    file = Path(sc_graph_serialization_file)
    assert (len(merged_sc_graph.nodes) == 41 + 130 - 10)
    assert (json.loads(file.read_text()) == json.loads(merged_sc_graph.json()))
    # see test_tf_import.py: how to import merged_sc_graph
    # test: parallelizer
    parallelizer = Parallelizer(merged_sc_graph)
    parallelizer.register_parallelism(DataParallelism(range(2)))
    parallelizer.run_parallelisms()
    # test: get tf runtime config for each sc_graph after parallelisms
    for sc_graph in parallelizer.graphs:
        config = tf_adapter.get_tf_runtime_config(sc_graph)
        config_file = os.path.join(os.path.dirname(__file__),
                                   "data", "SimpleCNN_model_desc.json")
        file = Path(config_file)
        assert (file.read_text() == json.dumps(config, indent=4,
                                               sort_keys=True))
    # see test_tf_import.py: how to export parallelizer.graphs as tf graph
    inserted_allreduce_file = \
        os.path.join(os.path.dirname(__file__),
                     "data", "SimpleCNNAllreduce.json")
    file = Path(inserted_allreduce_file)
    assert (file.read_text() == parallelizer.graphs[0].json())
    assert (len(parallelizer.graphs) == 2)
