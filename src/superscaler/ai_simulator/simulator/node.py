# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
Author: v-hohua
Version: 0.1
Update date: 12/16/2019

This module defines all attributes of node used in simulator module

v1.0 supported attributes:
Node metadata:
Attribute name      Type            Description
index               Int             ID of node
op                  String          Operation
name                String          Name of node

Execution info:
device              String          Device to run this node
execution_time      Float           Execution time of node
output_tensors      List of Tensor  Output tensors of node

Dependency info:
input_ids           List of int     Node ID of dataflow input
input_data_ids      List of int     Take which one of input node's output
                                    as the input data.
                                    In future need this to calculate data
                                    transfer overhead
dependency_ids      List of int     Node ID of dependency input
successor_ids       List of int     Node ID of successors

'''
import copy
from enum import Enum


class NodeMetadata():

    def __init__(self, index=0,
                 op='',
                 name='',
                 device_name='',
                 execution_time=0.0,
                 output_tensors=[],
                 input_ids=[],
                 dependency_ids=[],
                 successor_ids=[]
                 ):
        # ==============================
        # Attributes of node
        # ==============================

        # Int. The ID of node.
        self.index = index
        # String. The operation of node.
        self.op = op
        # String. The name of node.
        self.name = name

        # String, the device where the node is assigned.
        self.device_name = device_name
        # Float. The estimated execution time. In microsecond.
        self.execution_time = execution_time
        # Tensor, the list of output tensors
        self.output_tensors = output_tensors
        # ==============================
        # Attributes of edge
        # ==============================
        # These attributes is initialized by adapter outside this function.

        # List of int. Node of dataflow inputs
        self.input_ids = copy.deepcopy(input_ids)
        # List of int. Read which one of the input node's output.
        # Current version does not use this attribute
        # In most cases, this attribute is 0.
        # Read the first output of given node.
        # self.input_data_ids = []
        # List of int. Node of dependency inputs
        self.dependency_ids = copy.deepcopy(dependency_ids)
        # List of int. Node of successor nodes depends on this node.
        self.successor_ids = copy.deepcopy(successor_ids)

        # ==============================
        # Attributes for debugging and testing
        # ==============================
        # List of string. The raw input list in .pbtxt file.
        # The raw input name contains ':1' to indicate which output of a node
        # is the input.
        # self.raw_inputs = []

    def to_dict(self):
        return vars(self)

    def assign_from_dict(self, input_dict):
        for key in input_dict:
            setattr(self, key, input_dict[key])


class NodeStatus(Enum):
    waiting = 0
    executing = 1
    done = 2


class NodeException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def print_error_info(self):
        print(self.error_info)


'''
This class define a dynamic running node in simulator. While the NodeMetadata()
is a static definition of a node.
'''


class Node():
    def __init__(self, metadata, device):
        # The metadata of the node.
        self.__metadata = metadata
        # The status of a node.
        # 'waiting':       Not started.
        # 'executing':    Started, not finished. Only for asynchronized node.
        # 'done':       Finished.
        self.__status = NodeStatus.waiting
        # The number of dependency
        self.__remain_dependency_cnt = len(metadata.input_ids) + \
            len(metadata.dependency_ids)
        # The device runtime object that this node is running on
        self.__device = device
        # List of Node ref. Node of successor nodes depends on this node.
        self.__successor_nodes = []

        # Check the input_node and dependency_node
        input_ids_set = set(self.__metadata.input_ids)
        dependency_ids_set = set(self.__metadata.dependency_ids)
        if not len(dependency_ids_set) == len(self.__metadata.dependency_ids):
            raise NodeException(
                '[ERROR] Node initialization failure because dependency_ids '
                + 'has duplicate elements: %s' % self.__metadata.name)
        if not len(input_ids_set & dependency_ids_set) == 0:
            raise NodeException(
                '[ERROR] Node initialization failure because input_ids and '
                + 'dependency_ids has same elements: %s'
                % self.__metadata.name)

        # Check the device name
        if not self.__device.name() == self.__metadata.device_name:
            raise NodeException(
                '[ERROR] Node initialization failure because device_name not '
                + 'match: %s' % self.__metadata.name)

    def reset(self):
        metadata = self.__metadata
        self.__remain_dependency_cnt = len(metadata.input_ids) + \
            len(metadata.dependency_ids)
        self.__status = NodeStatus.waiting

    def is_ready(self):
        return self.__remain_dependency_cnt == 0

    def is_done(self):
        return self.__status == NodeStatus.done

    def get_index(self):
        return self.__metadata.index

    def get_op(self):
        return self.__metadata.op

    def get_name(self):
        return self.__metadata.name

    def get_device_name(self):
        return self.__metadata.device_name

    def get_execution_time(self):
        return self.__metadata.execution_time

    def set_execution_time(self, execution_time):
        self.__metadata.execution_time = execution_time

    def get_tensors(self):
        return self.__metadata.output_tensors

    def get_status(self):
        return self.__status

    def get_remain_dependency_cnt(self):
        return self.__remain_dependency_cnt

    def decrease_remain_dependency_cnt(self, cnt):
        if self.__remain_dependency_cnt >= cnt:
            self.__remain_dependency_cnt -= cnt
        else:
            raise NodeException(
                '[ERROR] node (%s) reduce an unexpected' % self.__metadata.name
                + ' remain_dependency_cnt (reduce: %s, current: %s)' %
                (cnt,
                 self.__remain_dependency_cnt))

    def renew_successor_nodes(self, node_list):
        self.__successor_nodes.clear()
        for suc_id in self.__metadata.successor_ids:
            self.__successor_nodes.append(node_list[suc_id])

    def get_successor_nodes(self):
        return self.__successor_nodes

    def execute(self, time_now):
        if not self.is_ready() or not self.__status == NodeStatus.waiting:
            raise NodeException(
                '[ERROR] Execute a non-ready node: %s' % self.__metadata.name)
        self.__status = NodeStatus.executing
        self.__device.enqueue_node(self, time_now)

    def finish(self):
        if not self.__status == NodeStatus.executing:
            raise NodeException(
                '[ERROR] Finish a non-execute node: %s' % self.__metadata.name)
        self.__status = NodeStatus.done
        self.__device.dequeue_node()
