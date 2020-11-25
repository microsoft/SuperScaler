# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from ctypes import cdll, c_int, byref, c_char_p


class Runtime(object):
    def __init__(self, plan_path, libsc_path):
        """A function that initializes Runtime.
        Args:
          plan_path: string specifying the path where communication plan is.
          libsc_path: string specifying the path where runtime library is.
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
        if not hasattr(self.libsc, "sc_get_device_id") or\
           not hasattr(self.libsc, "sc_get_host_id") or\
           not hasattr(self.libsc, "sc_get_world_size") or\
           not hasattr(self.libsc, "sc_finalize"):
            raise Exception("member function not exist")

    def get_sc_lib_path(self):
        """ A function that get the sc_lib_path. """
        return self.libsc_path

    def device_id(self):
        """ A function that returns thedevice_id of the calling process.

        Returns:
          An integer with the device_id of the calling process.
        """
        device_id = c_int()
        try:
            self.libsc.sc_get_device_id(byref(device_id))
            return device_id.value
        except BaseException:
            return None

    def host_id(self):
        """A function that returns the host_id of the calling process.

        Returns:
          An integer with the host_id of the calling process.
        """
        host_id = c_int()
        try:
            self.libsc.sc_get_host_id(byref(host_id))
            return host_id.value
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
