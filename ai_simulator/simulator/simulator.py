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
from .device import FIFODevice
import copy

RET_SIMULATION_FINISH = -1

class Simulator():
    def __init__(self, node_list):
        # List of NodeMetadata. All nodes in the graph
        self.__nodes_metadata = []
        # All nodes.
        self.__nodes = []
        # All devices.
        self.__devices = {}

        # The execution result of all nodes
        # Save node index and enqueue time
        self.__execution_result = []
        # Current simulation timestamp.
        self.__time_now = 0.0

        # Init all node metadat
        for i in range(len(node_list)):
            node = node_list[i]
            metadata = NodeMetadata(index = node.index, 
                op = node.op,
                name = node.name, 
                device_name = node.device, 
                execution_time = node.execution_time,
                input_ids = node.input_ids,
                dependency_ids = node.dependency_ids,
                successor_ids = node.successor_ids
                )
            self.__nodes_metadata.append(metadata)

        # Init devices list
        for node_metadata in self.__nodes_metadata:
            device_name = node_metadata.device_name
            if device_name not in self.__devices:
                self.__devices[device_name]=FIFODevice(device_name)
            new_node = Node(node_metadata, self.__devices[device_name])
            self.__nodes.append(new_node)
        
        # Init edges in nodes
        for node in self.__nodes:
            node.renew_successor_nodes(self.__nodes)

    '''
    Start all node in ready list.
    Only call once at the beginning of a simulation.
    '''
    def __start_all_ready_nodes(self):
        for node in self.__nodes:
            if node.is_ready():
                self.__start_node(node)

    '''
    Start to execute a node.Enqueue the node into device.
    The node will be marked as 'pending'. 

    @param exec_node:    Node ref. The node to execute.
    '''
    def __start_node(self, exec_node):
        node_id = exec_node.get_index()
        self.__execution_result.append([node_id, self.__time_now])
        exec_node.execute(self.__time_now)
        
        return

    def __find_earliest_complete_device(self):
        earliest_complete_time = RET_SIMULATION_FINISH
        earliest_device = None
        for device_name in self.__devices:
            device = self.__devices[device_name]
            if device.is_idle():
                continue
            device_complete_time = device.get_next_node()[1]
            if earliest_complete_time == RET_SIMULATION_FINISH or \
                    device_complete_time < earliest_complete_time:
                earliest_complete_time = device_complete_time
                earliest_device = device

        return earliest_complete_time, earliest_device

    '''
    Wait until any pending node is done. Get the timestamp.
    Mark the node as 'done'. Then dequeue it from device.
    Update all successor nodes' dependency counter.
    If a successor node is ready, start it.
    '''
    def __next_step(self):
        # Find the first completed node
        earliest_complete_time, earliest_device = \
            self.__find_earliest_complete_device()
        
        if earliest_complete_time == RET_SIMULATION_FINISH:
            return RET_SIMULATION_FINISH
        
        self.__time_now = earliest_complete_time
        earliest_node = earliest_device.get_next_node()[0]
        # Handle the node
        earliest_node.finish()
        # Handle successor nodes
        for suc_node in earliest_node.get_successor_nodes():
            suc_node.decrease_remain_dependency_cnt(1)
            if suc_node.is_ready():
                self.__start_node(suc_node)

        return earliest_complete_time
    
    '''
    Reset the simulator
    '''
    def reset(self):
        self.__time_now = 0.0
        self.__execution_result = []
        for node in self.__nodes:
            node.reset()

    '''
    Run the simulation
    '''
    def run(self):
        self.reset()
        finish_time = 0.0
        self.__start_all_ready_nodes()
        while(finish_time != RET_SIMULATION_FINISH):
            # Enqueue all ready nodes.
            
            # Wait until one node is done.
            finish_time = self.__next_step()
        return self.__time_now, self.__execution_result

    '''
    Get the all nodes
    '''
    def get_nodes(self):
        return self.__nodes



    


    
