from queue import PriorityQueue

from .utility import transfer_rate_to_bps, data_size_to_bit
from .device import Device


class FairSharingDeviceException(Exception):
    def __init__(self, error_info):
        super().__init__(error_info)

    def print_error_info(self):
        print(self.error_info)


class Flow():

    def __init__(self, node, time_now, available_bandwidth: float):
        '''Initialize a Flow.

        Args:
            available_bandwidth: a float representing tansfer rate in bps
        '''
        self.node = node
        self.__total_data_len = 0
        tensors = node.get_tensors()
        # Calculate total bytes of tensors
        for tensor in tensors:
            self.__total_data_len += tensor.get_bytes_size()
        # Store current status
        self.__remain_len\
            = data_size_to_bit(str(self.__total_data_len)+'B')  # Stored in bit
        self.__estimated_finish_time = time_now \
            + (self.__remain_len / available_bandwidth)
        # Store previous status.
        # There are two events that will change __prev_available_bandwidth
        # and __last_start_time: enqueue, dequeue.
        # This variables will change when there comes a new flow(enqueue)
        # or an old flow leaves(dequeue).
        # In a same FairSharingDevice, all flows have the same two variables
        self.__prev_available_bandwidth = available_bandwidth
        self.__last_start_time = time_now

    def get_estimated_finish_time(self):
        '''Return the estimated finish time of current flow'''
        return self.__estimated_finish_time

    def change_available_bandwidth(self, time_now,
                                   current_available_bandwidth: float):
        '''Update estimated_finish_time and other flow status

        Args:
            current_available_bandwidth: a float representing tansfer rate(bps)
        Calculate total executed bytes during [last_start_time, time_now],
        then update __remain_len and calculate new __estimated_finish_time
        '''
        executed_bits = (time_now - self.__last_start_time)\
            * self.__prev_available_bandwidth
        self.__remain_len -= executed_bits
        self.__estimated_finish_time = time_now\
            + self.__remain_len / current_available_bandwidth
        self.__last_start_time = time_now
        self.__prev_available_bandwidth = current_available_bandwidth

    def __lt__(self, other):
        '''Flow is ordered by estimated_finish_time'''
        if self.__estimated_finish_time < other.__estimated_finish_time:
            return True
        elif self.__estimated_finish_time == other.__estimated_finish_time:
            return self.node.get_index() < other.node.get_index()
        else:
            return False


class FairSharingDevice(Device):
    def __init__(self, name, capacity):
        '''Init a FairSharingDevice, create a Min-Heap of Flow

        Args:
            capacity: a string representing tansfer rate, e.g. '16Kibps'
                      or a float in bps, e.g. 16.0 == '16bps'
        '''
        super().__init__(name)
        self.__flows_priority_queue = PriorityQueue()  # A Min-Heap of Flow
        if isinstance(capacity, str):
            self.__capacity = transfer_rate_to_bps(capacity)  # Stored in bps
        elif isinstance(capacity, (int, float)):
            self.__capacity = float(capacity)
        else:
            raise ValueError("type of capacity is not valid")

    def is_idle(self):
        '''Return whether the device is idle.'''
        # Check whether there is node being executed
        return self.__flows_priority_queue.empty()

    def get_next_node(self):
        '''Return (next finish node, estimated finish time)'''
        if self.is_idle():
            return None, -1
        # self.__flows_priority_queue is a Min-Heap, so the first element
        # in __flows_priority_queue will be the next to be completed
        return (self.__flows_priority_queue.queue[0].node,
                self.__flows_priority_queue.queue[0]
                    .get_estimated_finish_time())

    def enqueue_node(self, node, time_now):
        '''Enqueue a node and update all the flows in this device

        The node will first be turned into a Flow, then push to the heap.
        '''
        # Calculate current bandwidth and create a new flow
        current_available_bandwidth = self.__capacity\
            / (self.__flows_priority_queue.qsize() + 1)
        new_flow = Flow(node, time_now, current_available_bandwidth)
        if not self.is_idle():
            # Update all flows
            self.__change_all_flows(time_now, current_available_bandwidth)
        self.__flows_priority_queue.put(new_flow)

    def dequeue_node(self):
        '''Dequeue the first flow in Min-Heap'''
        # Make sure the device is not idle
        if not self.is_idle():
            time_now = self.__flows_priority_queue.get()\
                            .get_estimated_finish_time()
            # If device is not idle after the next_node is finished
            if not self.is_idle():
                current_available_bandwidth = self.__capacity\
                    / self.__flows_priority_queue.qsize()
                # Update all flows
                self.__change_all_flows(time_now, current_available_bandwidth)

    def __change_all_flows(self, time_now,
                           current_available_bandwidth: float):
        '''Update the status of all flows'''
        # Before iteration, self.__flows_priority_queue is ordered
        for flow in self.__flows_priority_queue.queue:
            # The estimated_finish_time in flow will be changed during the call

            # The equations are:
            # flow.__remain_len = flow.__remain_len - executed_bytes
            # executed_bytes = (time_now - last_start_time) * prev_bandwidth
            # estimated_finish_time = time_now + remain_len / current_bandwidth

            # In FairSharingDevice, all flows have the same
            # prev/current_bandwidth and last_start_time/time_now.
            # So the order of estimated_finish_time is equivalent to the order
            # of flow.__remain_len.
            # Before the iteration, flow.__remain_len is ordered,
            # and as we can see, executed_bytes are same for all flows in this
            # device. Therefore, flow.__remain_len is still ordered after
            # this function call.
            flow.change_available_bandwidth(time_now,
                                            current_available_bandwidth)
        # After the iteration, self.__flows_priority_queue is still ordered
