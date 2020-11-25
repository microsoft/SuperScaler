# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import tempfile
from enum import Enum, unique


@unique
class LOGGER_NAME(Enum):
    SCALERGRAPH = "scalergraph"
    DEFAULT = "scalergraph.default"


_format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] \
- %(levelname)s: %(message)s')


def save(filepath=None,
         log_level=logging.DEBUG,
         logger_name=LOGGER_NAME.SCALERGRAPH):
    '''
    Argus:
        log_level: logging.DEBUG by default.
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    '''
    assert (logger_name in LOGGER_NAME)
    logger = logging.getLogger(logger_name.value)
    logger.setLevel(log_level)
    if filepath is None:
        _, filepath = tempfile.mkstemp(suffix='.log',
                                       prefix='ScalerGraph',
                                       text=True)
    filehandler = logging.FileHandler(filename=filepath, mode='w')
    filehandler.setFormatter(_format)
    logger.addHandler(filehandler)
    return filepath


def logger(logger_name=LOGGER_NAME.DEFAULT):
    assert (logger_name in LOGGER_NAME)
    logger = logging.getLogger(logger_name.value)
    handler = logging.StreamHandler()
    handler.setFormatter(_format)
    logger.addHandler(handler)
    return logger
