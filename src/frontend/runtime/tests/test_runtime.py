import os
import pytest
import multiprocessing
import traceback
from runtime import Runtime

father_path = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
lib_path = father_path + \
    "/lib/libtfadaptor.so"


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


class Process(multiprocessing.Process):
    # let the Process to throw out its own exceptions
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def is_cuda_available():
    """
        Check NVIDIA with nvidia-smi command
        Returning code 0 if no error, it means NVIDIA is installed
        Other codes mean not installed
    """
    code = os.system('nvidia-smi')
    return code == 0


def test_runtime():
    def func(rank):
        try:
            plan_path = 'plan_' + rank + '.json'
            plan_path = os.path.join(os.path.dirname(__file__),
                                     plan_path)

            # Check for init
            rt = Runtime(plan_path, lib_path)
            assert rt.device_id() == int(rank)
            assert rt.host_id() == 0
            assert rt.comm_world_size() == 2

        except BaseException:
            raise Exception

    # All backend codes must run on gpu environment with cuda support
    if is_cuda_available is True:
        p0 = Process(target=func, args=('0', ))
        p0.start()
        p0.join()
        p1 = Process(target=func, args=('1', ))
        p1.start()
        p1.join()

        if p0.exception:
            error, traceback = p0.exception
            raise ValueError("something wrong at p0.")

        if p1.exception:
            error, traceback = p1.exception
            raise ValueError("something wrong at p1.")
