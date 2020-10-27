import os
import pytest
from runtime import Runtime


def test_runtime_import():

    # Check None input
    with pytest.raises(Exception):
        rt = Runtime(None, None)
        rt.shutdown()
    # Check wrong path
    with pytest.raises(Exception):
        rt = Runtime("wrong Path", "wrong Path")
        rt.shutdown()

    # Init path location
    plan_path_0 = "plan_0.json"
    plan_path_0 = os.path.join(os.path.dirname(__file__),
                               plan_path_0)
    plan_path_1 = "plan_1.json"
    plan_path_1 = os.path.join(os.path.dirname(__file__),
                               plan_path_1)
    lib_path = "libtfadaptor.so"
    lib_path = os.path.join(os.path.dirname(__file__),
                            lib_path)

    # Check wrong lib_path
    with pytest.raises(Exception):
        rt = Runtime(plan_path_0, "wrong Path")
        rt.shutdown()
    # Check wrong plan_path
    with pytest.raises(Exception):
        rt = Runtime("wrong Path", lib_path)
        rt.shutdown()

    # Check fake library
    with pytest.raises(Exception):
        fake_lib_path = "libMySharedLib.so"
        fake_lib_path = os.path.join(os.path.dirname(__file__),
                                     fake_lib_path)
        rt = Runtime(plan_path_0, fake_lib_path)
        rt.shutdown()

    # #functional check
    # rt = Runtime(plan_path_0, lib_path)
    # assert rt.local_rank() == 0
    # assert rt.global_rank() == 0
    # assert rt.comm_world_size() == 2
