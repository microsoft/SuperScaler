'''
Author: v-hohua
Version: 0.1
Update date: 12/16/2019

Simulation Execution module of AI Simulator project.
This module will read execution graph, then simulate the execution.

v1.0 supported function:
Asynchronized execution on multiple devices.

v1.0 execute required node attributes:
Defined in node_define.py
'''

from .node import NodeMetadata
from .node import Node
from .device import Device
from .device_factory import DeviceFactory


RET_SIMULATION_FINISH = -1


class Simulator():
    def __init__(self, nodemetadata_list, device_info):
        '''Init Simulator with nodemetadata namedtuples and a device list

        Args:
            nodemetadata_list: a list of namedtuple, storing nodemetadata
            device_info: a list of tuple (device_type, spec_list) containing
                device info, or a list of class Device, storing all Device
        '''
        # List of NodeMetadata. All nodes in the graph
        self.__nodes_metadata = []
        # All nodes.
        self.__nodes = []
        # All devices.
        self.__devices = {}

        # The execution result of all nodes
        self.__execution_enqueue_time = []  # (node_index, node enqueue time)
        self.__execution_dequeue_time = []  # (node_index, node dequeue time)
        # Current simulation timestamp.
        self.__time_now = 0.0

        # Init all node metadata
        for i in range(len(nodemetadata_list)):
            node = nodemetadata_list[i]
            metadata = NodeMetadata(index=node.index,
                                    op=node.op,
                                    name=node.name,
                                    device_name=node.device_name,
                                    execution_time=node.execution_time,
                                    output_tensors=node.output_tensors,
                                    input_ids=node.input_ids,
                                    dependency_ids=node.dependency_ids,
                                    successor_ids=node.successor_ids
                                    )
            self.__nodes_metadata.append(metadata)

        # Init devices list
        self.__init_device(device_info)

        # Check the coherence of nodemetadata_list and device_info
        for node_metadata in self.__nodes_metadata:
            device_name = node_metadata.device_name
            if device_name not in self.__devices:
                raise TypeError(
                    device_name + " in nodemetadata_list doesn't exist on"
                    + " device_info")
            new_node = Node(node_metadata, self.__devices[device_name])
            self.__nodes.append(new_node)

        # Init edges in nodes
        for node in self.__nodes:
            node.renew_successor_nodes(self.__nodes)

    def __start_all_ready_nodes(self):
        '''Start all node in ready list.
        Only call once at the beginning of a simulation.
        '''
        for node in self.__nodes:
            if node.is_ready():
                self.__start_node(node)

    def __start_node(self, exec_node):
        '''Start to execute a node.Enqueue the node into device.
        The node will be marked as 'executing'.

        @param exec_node:    Node ref. The node to execute.
        '''
        node_id = exec_node.get_index()
        self.__execution_enqueue_time.append((node_id, self.__time_now))
        exec_node.execute(self.__time_now)

    def __find_earliest_complete_device(self):
        earliest_complete_time = RET_SIMULATION_FINISH
        earliest_device = None
        for device_name in self.__devices:
            device = self.__devices[device_name]
            if device.is_idle():
                continue
            device_complete_time = device.get_next_node()[1]
            if device_complete_time < earliest_complete_time or \
                    earliest_complete_time == RET_SIMULATION_FINISH:
                earliest_complete_time = device_complete_time
                earliest_device = device
        return earliest_complete_time, earliest_device

    def __next_step(self):
        '''Wait until any executing node is done. Get the timestamp.
        Mark the node as 'done'. Then dequeue it from device.
        Update all successor nodes' dependency counter.
        If a successor node is ready, start it.
        '''
        # Find the first completed node
        earliest_complete_time, earliest_device = \
            self.__find_earliest_complete_device()

        if earliest_complete_time == RET_SIMULATION_FINISH:
            return RET_SIMULATION_FINISH

        self.__time_now = earliest_complete_time
        earliest_node = earliest_device.get_next_node()[0]
        # Handle the node
        earliest_node.finish()
        self.__execution_dequeue_time.append((earliest_node.get_index(),
                                              self.__time_now))
        # Handle successor nodes
        for suc_node in earliest_node.get_successor_nodes():
            suc_node.decrease_remain_dependency_cnt(1)
            if suc_node.is_ready():
                self.__start_node(suc_node)

        return earliest_complete_time

    def reset(self):
        '''Reset the simulator'''
        self.__time_now = 0.0
        self.__execution_enqueue_time = []
        self.__execution_dequeue_time = []
        for node in self.__nodes:
            node.reset()

    def run(self):
        '''Run the simulation'''
        self.reset()
        finish_time = 0.0
        # Enqueue all ready nodes.
        self.__start_all_ready_nodes()
        while(finish_time != RET_SIMULATION_FINISH):
            # Wait until one node is done.
            finish_time = self.__next_step()
        return (self.__time_now,
                self.__execution_enqueue_time,
                self.__execution_dequeue_time)

    def get_nodes(self):
        '''Get the all nodes'''
        return self.__nodes

    def __init_device(self, device_info):
        '''Init self.__devices via device_info
        '''
        if not isinstance(device_info, list):
            raise ValueError("Input device_info should be a list.")
        if isinstance(device_info[0], Device):
            # device_info is a list of class Device
            for device in device_info:
                self.__devices[device.name()] = device
        elif isinstance(device_info[0], tuple):
            device_factory = DeviceFactory()
            # device_info is a list of (type, spec) tuple
            for device_type, spec_list in device_info:
                device = device_factory.generate_device(
                    device_type, *spec_list)
                self.__devices[device.name()] = device
        else:
            raise ValueError("Invalid input device_info parameter.")
