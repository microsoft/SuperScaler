# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.plan.resources.resource import Resource
from superscaler.plan_gen.plan.resources.hardware import ComputationHardware,\
    CPUHardware, GPUHardware


class Server(Resource):
    def __init__(self, name):
        '''Init a server with server name

        Args:
            name: string, /server/hostname1/
        '''
        hardware_type, server_name, *__ = name.split('/')[1:]
        if hardware_type != 'server':
            raise ValueError("Invalid Server name: %s" % name)
        self.__name = name
        self.__server_name = server_name
        self.__hardware = {}  # Computation Hardware dict {name: hardware_obj}

    def add_hardware(self, hardware):
        '''Add a computational hardware to the server

        Args:
            hardware: class ComputationHardware
        '''
        if not isinstance(hardware, ComputationHardware):
            raise ValueError("Can only add ComputationHardware to Server")
        hw_hostname, *_ = \
            ComputationHardware.get_computation_hardware_description(
                hardware.get_name()
            )
        if hw_hostname != self.__server_name:
            raise ValueError("Hardware doesn't belong to this Server")
        if hardware.get_name() in self.__hardware:
            raise ValueError("Cannot add two hardware with same name")
        self.__hardware[hardware.get_name()] = hardware

    def get_hardware_dict(self):
        '''Return a dict {hardware_name: hardware_obj}
        '''
        return self.__hardware

    def get_name(self):
        return self.__name

    def get_hardware_list_from_type(self, hardware_type):
        '''Return a list of (hardware_type) hardware

        Args:
            hardware_type: string, denoting the class type. e.g. 'CPU', 'GPU'
        Return:
            a list of type=hardware_type hardware
        '''
        # Use a dict instead of using dangerous eval() function
        valid_hardware_type = {'CPU': CPUHardware, 'CPUHardware': CPUHardware,
                               'GPU': GPUHardware, 'GPUHardware': GPUHardware}
        type_hardware_list = []
        for name, hardware in self.__hardware.items():
            if isinstance(hardware, valid_hardware_type[hardware_type]):
                type_hardware_list.append(hardware)
        return type_hardware_list

    def get_hardware_from_name(self, hardware_name):
        '''Return the hardware whose name is hardware_name, return None if not
        found

        Args:
            hardware_name: string, denoting the name of hardware
        '''
        if hardware_name in self.__hardware:
            return self.__hardware[hardware_name]
        return None
