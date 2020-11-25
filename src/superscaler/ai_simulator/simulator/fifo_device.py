# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.ai_simulator.simulator.device import Device


class FIFODevice(Device):
    def __init__(self, name):
        super().__init__(name)
        # The reference of enqueued nodes.
        self.__node_queue = []
        # Head pointer of node queue
        self.__queue_head = 0

    # Whether the device is idle.
    def is_idle(self):
        return self.__queue_head == len(self.__node_queue)

    # Return the ref of current executing node
    def get_next_node(self):
        return self.__node_queue[self.__queue_head], self.__next_finish_time

    # Enqueue a node to this device.
    # If the node is idle, update the head_end_time.
    def enqueue_node(self, node, time_now):
        if self.is_idle():
            self.__next_finish_time = time_now + node.get_execution_time()
        self.__node_queue.append(node)

    # Dequeue a node in this device.
    # If still has node in queue, reset head end time
    def dequeue_node(self):
        self.__queue_head += 1
        if not self.is_idle():
            head_node = self.__node_queue[self.__queue_head]
            next_node_timeuse = head_node.get_execution_time()
            self.__next_finish_time += next_node_timeuse
