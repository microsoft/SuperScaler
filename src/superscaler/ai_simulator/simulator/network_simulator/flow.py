# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.ai_simulator.simulator.utility import data_size_to_bit


class Flow():

    def __init__(self, node, time_now):
        '''Initialize a Flow.

        Args:
            node: class Node, a Send node
            time_now: float, current simulation time
        '''
        self.node = node
        self.__total_data_len = 0
        tensors = node.get_tensors()
        # Calculate total bytes of tensors
        for tensor in tensors:
            self.__total_data_len += tensor.get_bytes_size()
        # Store current status
        self.__remain_len = \
            data_size_to_bit(str(self.__total_data_len) + 'B')  # Stored in bit
        self.__estimated_finish_time = float('inf')
        self.__available_bandwidth = 0
        self.__last_start_time = time_now

    def get_estimated_finish_time(self):
        '''Return the estimated finish time of current flow'''
        return self.__estimated_finish_time

    def get_available_bandwidth(self):
        return self.__available_bandwidth

    def set_available_bandwidth(self, available_bandwidth, time_now):
        '''Set self.__current_available_bandwidth, then change
        estimated_finish_time and other flow status

        Args:
            current_available_bandwidth: a float representing tansfer rate(bps)
            time_now: current time of simulation
        Calculate total executed bytes during [last_start_time, time_now],
        then update __remain_len and calculate new __estimated_finish_time
        '''
        if available_bandwidth < 0:
            raise ValueError("Bandwidth should not be a negative number!")
        if time_now < self.__last_start_time:
            raise ValueError(
                "time_now should not be less than previous start time!")
        if time_now == self.__estimated_finish_time:
            # If time_now is the time that current flow will finish, then
            # directly assign remain_len to 0
            self.__remain_len = 0
            self.__last_start_time = time_now
            self.__available_bandwidth = available_bandwidth
        else:
            executed_bits = (time_now - self.__last_start_time) \
                * self.__available_bandwidth
            self.__remain_len -= executed_bits
            if available_bandwidth == 0:
                self.__estimated_finish_time = float('inf')
            else:
                self.__estimated_finish_time = time_now \
                    + self.__remain_len / available_bandwidth
            self.__last_start_time = time_now
            self.__available_bandwidth = available_bandwidth

    def __lt__(self, other):
        '''Flow is ordered by estimated_finish_time'''
        if self.__estimated_finish_time < other.__estimated_finish_time:
            return True
        elif self.__estimated_finish_time == other.__estimated_finish_time:
            return self.node.get_index() < other.node.get_index()
        else:
            return False
