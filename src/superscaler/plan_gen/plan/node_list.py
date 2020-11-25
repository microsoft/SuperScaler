# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class NodeList(object):

    def __init__(self, node_list=[]):
        self.node_list = []
        if isinstance(node_list, list):
            for node in node_list:
                new_node = Node(node)
                self.node_list.append(new_node)

    def __iter__(self):
        for node in self.node_list:
            yield node

    def __len__(self):
        return len(self.node_list)

    def to_json(self):
        ''' Transfer class info into json list
        '''
        output_json = []
        for node in self.node_list:
            output_json.append(node.to_json())
        return output_json

    def insert(self, index, node):
        if isinstance(node, Node):
            self.node_list.insert(index, node)

    def append(self, node):
        if isinstance(node, Node):
            self.node_list.append(node)

    def remove(self, node):
        if node in self.node_list:
            self.node_list.remove(node)

    def index(self, node):
        if node in self.node_list:
            return self.node_list.index(node)
        else:
            return None

    def get_node(self, index):
        if(isinstance(index, int) and index >= 0 and
           index < len(self.node_list)):
            return self.node_list[index]
        else:
            return None


class Node(object):

    def __init__(self, node_info={}):

        '''
        Args:
            node_info: <dict>
                name: <str> the name of node
                device: <str> the device name
                op: <str> the op name
                input: <list> the input dependency
                tensor_name: <str> the tensor_name of this tensor
                tensor_type: <int> the tensor_type of this tensor
                offset: <int> the offset for comm operator
                size: <int> the data_size for comm operator
                reduction: <str> the reduction including "", "sum" and "recv"
                target: <str> the target device
                related_op: <str> the related op name
                route_index: <int> the index of route from device to target
                route_type: <str> the type of route including "PCIE", "RDMA"
                execution_time: <flaot> the profiling execution time
        example:
            {
                "name": "test_Send_0",
                "device": "/server/hostname1/GPU/0/",
                "op": "Send",
                "input": ["test_Send_0"]
                "tensor_name": "test"
                "tensor_type": "DT_FLOAT"
                "offset": 50,
                "size": 50,
                "reduction":"sum",
                "target": "/server/hostname1/GPU/1/",
                "related_op": "test_Recv_0",
                "route_index": 0
                "route_type": "PCIE"
                "execution_time": 1.0
            }
        '''

        self.valid_node_info_type = {
            'name': str,
            'device': str,
            'op': str,
            'input': list,
            "output_shapes": list,
            "tensor_name": str,
            "tensor_type": str,
            "offset": int,
            "size": int,
            "reduction": str,
            "target": str,
            "related_op": str,
            "parent": str,
            "route_index": int,
            "route_type": str,
            "execution_time": float
        }

        if isinstance(node_info, dict):
            for key, info_type in self.valid_node_info_type.items():
                if key in node_info and isinstance(node_info[key], info_type):
                    setattr(self, key, node_info[key])
                else:
                    setattr(self, key, None)
        else:
            for key, _ in self.valid_node_info_type.items():
                setattr(self, key, None)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def to_json(self):
        ''' Transfer class info into json dict
        '''
        output_json = {}
        for key, info_type in self.valid_node_info_type.items():
            value = getattr(self, key, None)
            if isinstance(value, info_type):
                output_json[key] = value
        return output_json
