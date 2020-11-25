# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import humanreadable as hr
from superscaler.plan_gen.plan.resources.resource import Resource
from superscaler.plan_gen.plan.resources.link import Link


class Hardware(Resource):
    def __init__(self, name):
        '''Init a Hardware with name

        Args:
            name: string, format: /server/hostname/HardwareType/HardwareIndex/
                  e.g. /server/my_computer/GPU/0/
        '''
        self.__name = name
        self.__inbound_links = {}  # a dict, {src_hw_name: [multiple links]}
        self.__outbound_links = {}  # a dict, {dest_hw_name: [multiple links]}

    def add_link(self, link: Link):
        '''Add a new inbound link or outbound link

        Args:
            link: resources.link.Link
        '''
        # Check the link dest and src
        if link.source_hardware == link.dest_hardware:
            raise ValueError(
                "ComputationHardware cannot connect a link to self")
        # Add a new link
        if self.get_name() == link.source_hardware:
            # Add an outbound link
            dest_name = link.dest_hardware
            if dest_name not in self.__outbound_links:
                self.__outbound_links[dest_name] = [link]
            else:
                self.__outbound_links[dest_name].append(link)
        elif self.get_name() == link.dest_hardware:
            # Add an inbound link
            src_name = link.source_hardware
            if src_name not in self.__inbound_links:
                self.__inbound_links[src_name] = [link]
            else:
                self.__inbound_links[src_name].append(link)
        else:
            raise ValueError("Invalid link")

    def get_inbound_links(self):
        '''Return a dict, {src_hw_name: [multiple links]}
        '''
        return self.__inbound_links

    def get_name(self):
        return self.__name

    def get_outbound_links(self):
        '''Return a dict, {dest_hw_name: [multiple links]}
        '''
        return self.__outbound_links

    def __str__(self):
        '''For display purpose'''
        description = "Hardware name: " + self.get_name() + ":\n"
        description += "\tOutbound links: [\n\t\t"
        for dest_name, links in self.__outbound_links.items():
            for link in links:
                description += str(link.get_name())+"\n\t\t"
        description += "]\n"+"\tInbound Links: [\n\t\t"
        for src_name, links in self.__inbound_links.items():
            for link in links:
                description += str(link.get_name())+"\n\t\t"
        description += "]"
        return description

    def to_dict(self):
        '''Return the dict represtation of Hardware essential data
        '''
        return {'name': self.__name}


class ComputationHardware(Hardware):
    def __init__(self, name, performance='0bps'):
        '''Init a Hardware with name

        Args:
            name: string, format: /server/hostname/HardwareType/HardwareIndex/
                  e.g. /server/my_computer/GPU/0/
            performance: a string representing calculating performance.
                         e.g. '16Gibps'
        '''
        super().__init__(name)
        self.__performance = \
            hr.BitPerSecond(performance).kibi_bps*1024  # Stored in bps

    def get_performance(self):
        '''Return the performance in bps
        '''
        return self.__performance

    @staticmethod
    def get_computation_hardware_description(name_str):
        '''Parse name_str, return (hostname, ComputationType,
        hardware_index, extra_info)

        Args:
            name_str: string, format: /server/hostname/ComputationType/Index/..
        If name_str is not valid, a ValueError will be raised.
        '''
        hardware_description = name_str.split('/')
        if(len(hardware_description) < 6):
            raise ValueError("Invalid ComputationHardware name: %s"
                             % name_str)
        if(hardware_description[1] != 'server'):
            raise ValueError("Invalid ComputationHardware name: %s"
                             % name_str)
        host_name, hardware_type, hardware_index, *extra_info = \
            hardware_description[2:]
        return host_name, hardware_type, hardware_index, extra_info

    def to_dict(self):
        '''Return the dict represtation of ComputationHardware essential data
        '''
        return dict({'performance': str(self.__performance)+'bps'},
                    **super().to_dict())


class CPUHardware(ComputationHardware):
    def __init__(self, name, performance='0bps'):
        '''Init a CPU with name and performance.

        name example: /server/hostname/CPU/0/
        '''
        # Check if name is valid
        _, hardware_type, *__ = \
            ComputationHardware.get_computation_hardware_description(name)
        if hardware_type != "CPU":
            raise ValueError("Invalid CPUHardware name: %s" % name)
        super().__init__(name, performance)


class GPUHardware(ComputationHardware):

    def __init__(self, name, performance='0bps'):
        '''Init a GPU with name and performance.

        name example: /server/hostname/GPU/0/
        '''
        # Check if name is valid
        _, hardware_type, *__ = \
            ComputationHardware.get_computation_hardware_description(name)
        if hardware_type != "GPU":
            raise ValueError("Invalid CPUHardware name: %s" % name)
        super().__init__(name, performance)


class NetworkSwitchHardware(Hardware):
    def __init__(self, name):
        '''Init a network switch with switch name

        Args:
            name: string, /switch/switchname/
        '''
        # Check if name is valid
        hardware_type, switch_name, *__ = name.split('/')[1:]
        if hardware_type != 'switch':
            raise ValueError(
                "Invalid NetworkSwitch name: %s" % name)
        super().__init__(name)
