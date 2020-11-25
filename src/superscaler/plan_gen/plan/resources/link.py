# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import humanreadable as hr
from superscaler.plan_gen.plan.resources.resource import Resource


class Link(Resource):
    def __init__(self, link_id, source_hardware, dest_hardware,
                 capacity='0bps', latency='0s', scheduler='FIFO',
                 link_type="BasicType"):
        '''Init an unidirectional link with information

        Args:
            link_id: int, the identity of current Link
            source_hardware: string, name of source hardware
            dest_hardware: string, name of destation hardware
            capacity: string, the capacity of this link
            latency: string, the propagation_latency of the link
            scheduler: string, the scheduling algorithm of the link
        '''
        self.__link_id = link_id
        self.__source_hardware = source_hardware
        self.__dest_hardware = dest_hardware
        # capacity is stored in bps
        self.__capacity = hr.BitPerSecond(capacity).kibi_bps*1024
        self.__latency = latency
        self.__scheduler = scheduler
        self.__link_type = link_type

    def get_name(self):
        '''Return link name:
            #link#type#source_hardware_name#dest_hardware_name#...
        '''
        return "#link#{0}#{1}#{2}".format(
            self.link_type,
            self.source_hardware,
            self.dest_hardware)

    def to_dict(self):
        '''Return a dict containing all essential data
        '''
        info_dict = {
            'source_name': self.__source_hardware,
            'dest_name': self.__dest_hardware,
            'capacity': str(self.__capacity) + 'bps',
            'latency': self.__latency,
            'scheduler': self.__scheduler,
            'link_type': self.__link_type,
            'link_id': self.__link_id
        }
        return info_dict

    @property
    def source_hardware(self):
        return self.__source_hardware

    @property
    def dest_hardware(self):
        return self.__dest_hardware

    @property
    def capacity(self):
        return self.__capacity

    @property
    def latency(self):
        return self.__latency

    @property
    def scheduler(self):
        return self.__scheduler

    @property
    def link_type(self):
        return self.__link_type

    @property
    def link_id(self):
        return self.__link_id


class PCIE(Link):
    def __init__(self, link_id, source_hardware, dest_hardware,
                 capacity='0bps', latency='0s', scheduler='FIFO'):
        super().__init__(link_id, source_hardware, dest_hardware,
                         capacity, latency, scheduler, "PCIE")


class RDMA(Link):
    def __init__(self, link_id, source_hardware, dest_hardware,
                 capacity='0bps', latency='0s', scheduler='FIFO'):
        super().__init__(link_id, source_hardware, dest_hardware,
                         capacity, latency, scheduler, "RDMA")
