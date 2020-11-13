import re
import os
import json
import copy
from superscaler.plan_gen.plan.adapter.adapter import Adapter
from superscaler.plan_gen.plan.node_list import NodeList


class SuperScalerAdapter(Adapter):
    """Interface for adapting generated plan for the Superscaler."""

    def __init__(self):
        super().__init__()
        self.__node_list = []

    def set_plan(self, node_list):
        ''' Set self.__node_list as list format generated from a node_list
        Args:
            node_list: NodeList
        '''
        if isinstance(node_list, NodeList):
            self.__node_list = node_list.to_json()
        elif isinstance(node_list, list):
            self.__node_list = node_list
        else:
            self.__node_list = []

    def dump_plan(self, dump_path, dump_name='superscaler'):
        """ Dump plan into dump_path with dump_name.

        Args:
            dump_path: string
            dump_name: string
        """

        multi_comm_node_dict = self.adapt_plan()

        for id_ in range(len(multi_comm_node_dict)):
            path = os.path.join(dump_path, dump_name + str(id_) + ".json")
            json.dump(multi_comm_node_dict[id_],
                      open(path, 'w'),
                      indent=4,
                      sort_keys=True)

    def adapt_plan(self):
        """ Adapt plan includes four steps
            1, extract all communication nodes from self.__node_list
            2, introduce dependency on all communication nodes
            3, split device info as host_id/device_type/device_id
            4, differentiate nodelist based on its device mapping

        Return:
            multi_comm_node_dict: dict
        """

        comm_node_list = self.__extract_comm_nodes()
        self.__create_index_dependency(comm_node_list)
        self.__split_device_info(comm_node_list)
        multi_comm_node_dict =\
            self.__differentiate_node_list(comm_node_list)

        return multi_comm_node_dict

    def __extract_comm_nodes(self):
        """ Extract communication nodes of Send, Recv and Allreduce
            from self.__node_list and group them into the comm_node_list.
        """

        comm_ops = ['Send', 'Recv', 'Allreduce']

        comm_node_list = []
        node_list = copy.deepcopy(self.__node_list)

        for node in node_list:
            if 'op' in node and node['op'] in comm_ops:
                comm_node_list.append(node)

        return comm_node_list

    def __create_index_dependency(self, node_list):
        ''' Introduce index attr into node_list, then convert
            input name dependency into index dependency.
            key is also introduced here.
        Args:
            node_list: list
        '''

        ''' node_book: Use a dict to stroe index for node_index
            Key: pair(name:string, device:string)
                 (name, device) pair uniquely identify a node
            value: index:int
        '''
        node_book = {}
        key_book = {}
        key_value = 0

        # Introduce index attr to each node, and use node_book and
        # parent_book to record them
        for index, node in enumerate(node_list):

            # Introduce index attr
            node['index'] = index

            # Record all nodes on node_book
            name = node['name']
            device = node['device']
            node_book[(name, device)] = index

        # Generate input dependency for each node by books
        for node in node_list:

            # Dependency is presented as id
            input_ids = []
            device = node['device']

            # Transfer name dependency to index dependency
            if 'input' in node:
                for input_name in node['input']:
                    if (input_name, device) in node_book:
                        input_ids.append(node_book[(input_name, device)])
                node.pop('input')
            node['input_ids'] = input_ids

        # Generate key for each generated node, where related send/recv
        # nodes share same key
        # eg. Send <-> Recv
        for node in node_list:
            if 'related_op' in node:
                name = node['name']
                device = node['device']
                related_op = node['related_op']
                target = node['target']

                key = (name, device, related_op, target)
                key_opponent = (related_op, target, name, device)

                if key not in key_book or key_opponent not in key_book:
                    key_book[key] = key_value
                    key_book[key_opponent] = key_value
                    node['key'] = node['parent'] + '_' + str(key_value)
                    key_value += 1
                else:
                    node['key'] = node['parent'] + '_' + str(key_book[key])

                node.pop('related_op')

    def __split_device_info(self, node_list):
        """ Split device info from /server/hostname1/GPU/0 into
            host_id:1, device_type:GPU, device_id:0.
            This part works for Superscaler's information requirement.
        Args:
            node_list: list
        """

        for node in node_list:
            if 'device' in node:
                device_info = node['device']
                splited_infos = str.split(device_info, '/')
                node['host_id'] = re.sub("\\D", "", splited_infos[2])
                node['device_type'] = splited_infos[3]
                node['device_id'] = splited_infos[4]
            if 'target' in node:
                target_info = node['target']
                splited_infos = str.split(target_info, '/')
                node['target_host_id'] = re.sub("\\D", "", splited_infos[2])
                node['target_device_type'] = splited_infos[3]
                node['target_device_id'] = splited_infos[4]

    def __differentiate_node_list(self, node_list):
        """ Differentiate a complete node_list based on device
            This part works for Superscaler's file structre requirement.
        Args:
            node_list: list
        Return:
            multi_node_list: list
        """

        # Record all devices of node_list, for each device a node_dict
        # is created with device infos and peer infos
        devices = []
        multi_node_dict = []

        for node in node_list:
            if 'device' in node:
                if node['device'] not in devices:
                    devices.append(node['device'])
                    multi_node_dict.append(
                        {
                         'host_id': node['host_id'],
                         'device_type': node['device_type'],
                         'device_id': node['device_id'],
                         'num_peers': 1,
                         'peer_device_names': [node['device']],
                         'tasks': []
                        })

                # pop unused device infos
                node.pop('host_id')
                node.pop('device_type')
                node.pop('device_id')

        # differentiate node_list on each node_dict
        for node in node_list:
            node_dict = multi_node_dict[devices.index(node['device'])]
            node_dict['tasks'].append(node)

            # Count the peers for communication
            if 'target' in node and \
               node['target'] not in node_dict['peer_device_names']:
                node_dict['num_peers'] += 1
                node_dict['peer_device_names'].append(node['target'])

            # pop unused device infos
            node.pop('device')
            node.pop('target')

        # Using str format
        for node_dict in multi_node_dict:
            node_dict['num_peers'] = str(node_dict['num_peers'])

        return multi_node_dict
