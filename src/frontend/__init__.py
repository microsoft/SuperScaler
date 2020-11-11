import os
import sys
if "tensorflow" in sys.modules or "tf" in sys.modules:
    raise Exception("import superscaler before tensorflow")
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "4"
from frontend.tensorflow import tensorflow

__all__ = ['tensorflow']

