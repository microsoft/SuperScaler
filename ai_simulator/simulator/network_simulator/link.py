from .flow import Flow
from simulator.utility import transfer_rate_to_bps


class Link():
    def __init__(self, source_name, dest_name,
                 capacity='0bps', latency='0s'):
        '''Init an unidirectional link with information

        Args:
            source_name: string, name of source device
            dest_name: string, name of destation device
            capacity: string, the capacity of this link
            latency: string, the propagation_latency of the link, reserved for
                future usage
        '''
        self.__source_name = source_name
        self.__dest_name = dest_name
        # capacity is stored in bps
        self.__capacity = transfer_rate_to_bps(capacity)
        self.__latency = latency
        self.__flows = []

    def add_flow(self, flow: Flow):
        '''Add a new flow to this link
        '''
        self.__flows.append(flow)

    def delete_flow(self, flow: Flow):
        '''Delete specific flow from this link
        '''
        self.__flows.remove(flow)

    @property
    def source_name(self):
        return self.__source_name

    @property
    def dest_name(self):
        return self.__dest_name

    @property
    def capacity(self):
        return self.__capacity

    @property
    def latency(self):
        # This property currently only reserved for future usage
        return self.__latency

    @property
    def flows(self):
        return self.__flows
