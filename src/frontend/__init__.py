import os
import tempfile
import sys
if "tensorflow" in sys.modules or "tf" in sys.modules:
    raise Exception("import superscaler before tensorflow")
WORKDIR_HANDLER = tempfile.TemporaryDirectory()
os.environ["TF_DUMP_GRAPH_PREFIX"] = WORKDIR_HANDLER.name
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "4"
from frontend.tensorflow import tensorflow

__all__ = ['tensorflow']

