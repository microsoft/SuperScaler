import os
from ctypes import cdll, c_int, byref, c_char_p


class Runtime(object):
    def __init__(self, plan_path, libsc_path):
        """A function that initializes Runtime.
        Args:
          plan_path: string specifying the path where communication plan is.
        """
        # Check for the existence of plan_path
        if not os.path.exists(plan_path):
            raise Exception("plan_path: %s is not existed!" % (plan_path))
        self.plan_path = plan_path

        # Check for the existence of libsc_path
        if not os.path.exists(libsc_path):
            raise Exception("libsc_path: %s is not existed!" % (libsc_path))
        self.libsc_path = libsc_path

        # Check for the loading of libsc_path
        try:
            libsc = cdll.LoadLibrary(libsc_path)
        except BaseException:
            raise Exception("libsc_path: %s is not loaded by ctype!" %
                            (libsc_path))
        self.libsc = libsc

        # Check for the init of libsc
        try:
            self.libsc.sc_init(c_char_p(self.plan_path.encode('utf-8')))
        except BaseException:
            raise Exception("libsc init failed")

        # Check for the member function of libsc
        if not hasattr(self.libsc, "sc_get_local_rank") or\
           not hasattr(self.libsc, "sc_get_global_rank") or\
           not hasattr(self.libsc, "sc_get_world_size") or\
           not hasattr(self.libsc, "sc_finalize"):
            raise Exception("member function not exist")

    def get_sc_lib_path(self):
        """ A function that get the sc_lib_path. """
        return self.libsc_path

    def local_rank(self):
        """ A function that returns the local rank of the calling process,
        within the node that it is running on. If there are seven processes
        running on a node, their local ranks will be zero through six.

        Returns:
          An integer with the local rank of the calling process.
        """
        lrank = c_int()
        try:
            self.libsc.sc_get_local_rank(byref(lrank))
            return lrank.value
        except BaseException:
            return None

    def global_rank(self):
        """A function that returns the global rank of the calling process.

        Returns:
          An integer with the global rank of the calling process.
        """
        grank = c_int()
        try:
            self.libsc.sc_get_global_rank(byref(grank))
            return grank.value
        except BaseException:
            return None

    def comm_world_size(self):
        """A function that returns the number of processes.

        Returns:
          An integer containing the number of processes.
        """
        gsze = c_int()
        try:
            self.libsc.sc_get_world_size(byref(gsze))
            return gsze.value
        except BaseException:
            return None

    def shutdown(self):
        """ A function that shuts runtime down. """
        try:
            self.libsc.sc_finalize()
        except BaseException:
            raise Exception("shutdown failed")

    def __del__(self):
        self.shutdown()
        del self.libsc
