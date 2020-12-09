# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from pathlib import Path
from superscaler.scaler_graph.parallelization.parallelism import \
     DataParallelism
from superscaler.scaler_graph.parallelization.parallelizer import Parallelizer
from superscaler.scaler_graph.IR.conversion import tf_adapter


def test_dataparallelism():
    # test: tf merged import
    tf_pbtxt_path = os.path.join(os.path.dirname(__file__), "data/simple_cnn",
                                 "SimpleCNN.pbtxt")
    tf_runtime_config = {
        "feeds": [],
        "fetches": ["cross_entropy:0"],
        "inits": ["init"],
        "targets": ["GradientDescent"]
    }
    merged_sc_graph = tf_adapter.import_graph_from_tf_pbtxts([tf_pbtxt_path],
                                                             tf_runtime_config)
    # see test_tf_adapter.py: how to import merged_sc_graph
    # test: parallelizer
    parallelizer = Parallelizer(merged_sc_graph)
    parallelizer.register_parallelism(DataParallelism(range(2)))
    parallelizer.run_parallelisms()
    # test: get tf runtime config for each sc_graph after parallelisms
    for sc_graph in parallelizer.graphs:
        config = tf_adapter.get_tf_runtime_config(sc_graph)
        config_file = os.path.join(os.path.dirname(__file__),
                                   "data/simple_cnn",
                                   "SimpleCNN_model_desc.json")
        file = Path(config_file)
        assert (file.read_text() == json.dumps(config,
                                               indent=4,
                                               sort_keys=True))
        # see test_tf_import.py: how to export parallelizer.graphs as tf graph

        inserted_allreduce_file = os.path.join(os.path.dirname(__file__),
                                               "data/simple_cnn",
                                               "SimpleCNNAllreduce.json")
        file = Path(inserted_allreduce_file)
        assert (file.read_text() == sc_graph.json())
    assert (len(parallelizer.graphs) == 2)
