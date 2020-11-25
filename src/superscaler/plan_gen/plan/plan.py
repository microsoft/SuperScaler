# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from superscaler.plan_gen.plan.node_list import NodeList
from superscaler.plan_gen.plan.node_list import Node


class Plan(object):
    """ An class that generate optimized plan from nodelist.
        target for nodes whose ops equal to plan_type
    """

    def __init__(self,
                 plan_name='Default_Plan',
                 plan_type='Default'):
        ''' Init a plan with name and type
        Args:
            name: string, e.g. Default_Plan
            type: string, e.g. Default
        '''
        self.__plan_name = plan_name
        self.__plan_type = plan_type

        # The major compoment of the plan class
        self.__node_list = NodeList()

    def get_plan_type(self):
        ''' Get the plan_type attr
        '''
        return self.__plan_type

    def get_plan_name(self):
        ''' Get the plan_name attr
        '''
        return self.__plan_name

    def get_plan_info(self):
        ''' Get the plan_type and plan_name attrs
        '''
        return self.get_plan_type(), self.get_plan_name()

    def reset_node_list(self, node_list):
        ''' Set self.__node_list
        Args:
            node_list: list
        '''
        if not isinstance(node_list, list):
            self.__node_list = None
        else:
            self.__node_list = NodeList(node_list)

    def generate_plan(self):
        ''' Main function, will be overloadding by child class
        '''
        return self._get_node_list()

    def _get_node_list(self):
        ''' Get node list
        '''
        return self.__node_list

    def _generate_node(self,
                       node_index,
                       node_name,
                       input_name,
                       target_name,
                       op,
                       reduction,
                       offset,
                       size,
                       target,
                       node_info):
        ''' generate a node and insert it into node list
        Args:
            node_index: <int> insert generated to the index of node list
            node_name: <str> the name of generated node
            input_name: <str>/<None> the additional input dependency
            target_name: <str> the related op name
            op: <str> the op of node
            reduction: <str> the reduction including "", "sum" and "recv"
            offset: <int> the offset for comm operator
            size: <int> the data_size for comm operator
            target: <str> the target device
            node_info: <dict> a dict with infomation for generated node
        return:
            generated_node: <dict>
        '''
        generated_node = {'name': node_name,
                          'offset': offset,
                          'size': size,
                          'op': op,
                          'reduction': reduction,
                          'target': target,
                          'related_op': target_name,
                          'device': node_info.device,
                          'output_shapes': node_info.output_shapes,
                          'tensor_name': node_info.tensor_name,
                          'tensor_type': node_info.tensor_type,
                          'parent': node_info.name,
                          'input': copy.deepcopy(node_info.input)}

        if input_name is not None:
            generated_node['input'].append(input_name)

        generated_node = Node(generated_node)
        self.__node_list.insert(node_index, generated_node)

        return generated_node
