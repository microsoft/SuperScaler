from .adapter import Adapter
from ..node_list import NodeList


class AISimulatorAdaptor(Adapter):
    """Interface for adapting generated plan for the AI_Simulator."""

    def __init__(self):
        super().__init__()
        self.__plan = []

    def set_plan(self, node_list):
        ''' Set self.__plan from a node_list, return False if input is invalid

        Args:
            node_list: list
        '''
        if(not isinstance(node_list, NodeList) and
           not isinstance(node_list, list)):
            self.__plan = None
            return False
        else:
            if isinstance(node_list, NodeList):
                self.__plan = node_list.to_json()
            elif isinstance(node_list, list):
                self.__plan = node_list
            if not self.__check_plan_attr():
                self.__plan = None
                return False
            if self.__parse_plan():
                return True
            else:
                self.__plan = None
                return False

    def get_plan(self):
        return self.__plan

    def __check_plan_attr(self):
        ''' Check where the input nodes follow a correct format.
        Return False if plan is invalid
        '''
        # The computational nodes must have these 4 attributes
        essential_compute_attr_type = {
            'device': str, 'name': str, 'op': str, 'input': list}
        # The communication nodes must have these 8 attributes
        essential_comm_attr_type = dict(
            {
                'target': str,
                'route_index': int,
                'output_shapes': list,
                'tensor_type': str
            },
            **essential_compute_attr_type
        )
        # These are the valid communication node op's name
        valid_comm_node_op_name = ['Send', 'Recv', 'Allreduce']
        for node in self.__plan:
            if 'op' not in node:
                return False
            elif node['op'] not in valid_comm_node_op_name:
                # A computational Node
                for attr, var_type in essential_compute_attr_type.items():
                    if attr not in node:
                        return False
                    if not isinstance(node[attr], var_type):
                        return False
            else:
                # A communication Node
                for attr, var_type in essential_comm_attr_type.items():
                    if attr not in node:
                        return False
                    if not isinstance(node[attr], var_type):
                        return False
        return True

    def __parse_plan(self):
        """ Parsing plan includes two steps, return False if failed
            1. create index dependency for each node with new arguments of
            index, input_ids, related_id, successor_ids
            2. add execution time using Profiler
            2. add empty arguments for AISimulator
        """
        # create dependency for each node
        if not self.__create_index_dependency():
            return False
        # add estimated execution_time for each node
        elif not self.__add_execution_time():
            return False
        else:
            return True

    def __create_index_dependency(self):
        '''
        convert input name dependency into index dependency,
        return False if failed
        '''

        ''' node_book: Use a dict to store index for node_index
            Key: pair(name:string, device:string)
                 (name, device) pair uniquely identify a node
            value: index:int
        '''
        node_book = {}

        ''' parent_book: Use a dict to store index for generated nodes
                their parent identify the name of original node
            Key: pair(parent:string, device:string)
                 (parent, device) pair uniquely identify a parent node
            Value: list(index:int)
        '''
        parent_book = {}

        # Introduce index attr to each node, and use node_book and
        # parent_book to record them
        for index, node in enumerate(self.__plan):

            # Introduce index attr
            node['index'] = index

            # Record all nodes on node_book
            name = node['name']
            device = node['device']
            node_book[(name, device)] = index

            # Record generated nodes on parent_book
            if 'parent' in node:
                parent = node['parent']
                if (parent, device) not in parent_book:
                    parent_book[(parent, device)] = [index]
                else:
                    parent_book[(parent, device)].append(index)
                node.pop('parent')

        # Generate input dependency for each node by books
        for node in self.__plan:

            # Dependency is presented as id
            input_ids = []
            device = node['device']

            # Transfer name dependency to index dependency
            if 'input' in node:
                for input_name in node['input']:
                    if (input_name, device) in node_book:
                        input_ids.append(node_book[(input_name, device)])
                    elif (input_name, device) in parent_book:
                        for id_ in parent_book[(input_name, device)]:
                            input_ids.append(id_)
                    else:
                        return False
            node.pop('input')
            node['input_ids'] = input_ids

        # Generate related id for each generated node
        # eg. Send <-> Recv
        for node in self.__plan:
            if 'related_op' in node:
                related_op = node['related_op']
                target = node['target']
                if (related_op, target) not in node_book:
                    return False
                node['related_id'] = node_book[(related_op, target)]
                node.pop('related_op')

        # Generate successor ids for each node from input_ids
        for node in self.__plan:
            node['successor_ids'] = []

        for node in self.__plan:
            for input_id in node['input_ids']:
                self.__plan[input_id]['successor_ids'].append(node['index'])

        # Generate dependency_ids for each node
        for node in self.__plan:
            # dependency_ids: argument for Recv node to point Send node
            # For other nodes the dependency_ids is empty list
            if node['op'] == 'Recv':
                node['dependency_ids'] = [node['related_id']]
                self.__plan[node['related_id']]['successor_ids'].append(
                    node['index']
                )
            else:
                node['dependency_ids'] = []

        return True

    def __add_execution_time(self):
        '''Add the execution time to nodes
        '''
        # TODO: In future, this function will call Profiler to get the
        # estimated execution time
        for node in self.__plan:
            # execution_time: argument for AISimulator to fill
            if 'execution_time' not in node:
                node['execution_time'] = 0.0
        return True
