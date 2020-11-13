import os
import pytest
import subprocess
import tensorflow as tf
from superscaler.runtime.tensorflow.runtime import TFRuntime


def is_gpu_available():
    """
        Check NVIDIA with nvidia-smi command
        Returning code == 0 and count > 0, it means NVIDIA is installed
        and GPU is available for running
        Other means not installed
    """
    code = os.system('nvidia-smi')
    if code == 0:
        cmd = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
        count = subprocess.check_output(cmd, shell=True)
        return int(count) > 0
    else:
        return False


def test_tfruntime_import():

    # Check None input
    with pytest.raises(Exception):
        rt = TFRuntime(None, None, None, None)
        rt.shutdown()
    # Check wrong path
    with pytest.raises(Exception):
        rt = TFRuntime("wrong Path", "wrong Path", "wrong Path", "wrong Path")
        rt.shutdown()


def test_tfruntime():
    # All backend codes must run on gpu environment with cuda support
    if is_gpu_available() is True:

        # Init path location
        graph_path = os.path.join(os.path.dirname(__file__),
                                  "data/graph.pbtxt")
        graph_config_path = os.path.join(os.path.dirname(__file__),
                                         "data/model_desc.json")
        plan_path = os.path.join(os.path.dirname(__file__),
                                 "data/plan.json")
        father_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../.."))
        lib_path = father_path + \
            "/lib/libtfadaptor.so"

        # Init TFRuntime
        rt = TFRuntime(graph_path, graph_config_path, plan_path, lib_path)

        # Check TFRuntime function
        inits = rt.inits
        for init in inits:
            assert(isinstance(init, tf.Operation))
        graph = rt.graph
        assert(isinstance(graph, tf.Graph))
        feeds = rt.feeds
        for feed in feeds:
            assert(isinstance(feed, str))
        fetches = rt.fetches
        for fetch in fetches:
            assert(isinstance(fetch, tf.Tensor))
        targets = rt.targets
        for target in targets:
            assert(isinstance(target, tf.Tensor))
