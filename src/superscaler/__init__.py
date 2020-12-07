# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import tempfile
import sys
if "tensorflow" in sys.modules or "tf" in sys.modules:
    raise Exception("import superscaler before tensorflow")
WORKDIR_HANDLER = tempfile.TemporaryDirectory()
os.environ["TF_DUMP_GRAPH_PREFIX"] = WORKDIR_HANDLER.name
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "4"
os.environ["SUPERSCLAR_PATH"] = os.path.dirname(__file__)
from superscaler.tensorflow import tensorflow  # noqa: 402
from superscaler.nnfusion import generate_data_parallelism_plan # noqa: 402

__all__ = ['tensorflow', 'generate_data_parallelism_plan']
