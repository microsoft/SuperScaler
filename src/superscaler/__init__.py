# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from superscaler.tensorflow import tensorflow
from superscaler.tensorflow import TFRecordDataset
from superscaler.nnfusion import generate_data_parallelism_plan
os.environ["SUPERSCLAR_PATH"] = os.path.dirname(__file__)

__all__ = ['tensorflow', 'generate_data_parallelism_plan', 'TFRecordDataset']
