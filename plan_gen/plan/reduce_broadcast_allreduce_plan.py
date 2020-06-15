import copy
from plan.plan import AllreducePlan


class ReduceBroadcastAllreducePlan(AllreducePlan):

    def __init__(self, plan_name="Reduce_Broadcast_Allreduce_Plan"):
        super().__init__(plan_name=plan_name)

    def generate_plan(self):
        '''
        Generating plan includes three step:
        1. find all allreudce nodes as allreduce_node_list
        2. For a specific node, find its related nodes as endpoints
        3. separate all allreduce node to Reduce_Broadcast allreduce nodes
        '''

        # Check input plan
        if not isinstance(self._get_plan(), list):
            return None

        # record original node_list for plan generator
        node_list_ref = copy.deepcopy(self._get_plan())

        allreduce_node_list = self.find_all_allreduce_nodes(node_list_ref)
        for node in allreduce_node_list:
            endpoint = self.find_endpoints(node, node_list_ref)
            self.separate_allreduce_node(node,
                                         endpoint)

        node_list_ref.clear()
        return self._get_plan()

    def find_all_allreduce_nodes(self, node_list):
        ''' return a allreduce_node_list with allreduce op
        Args:
            node_list: list, the input nodelist
        '''
        allreduce_node_list = []
        for node in node_list:
            if node['op'] == self.get_plan_type():
                allreduce_node_list.append(node)
        return allreduce_node_list

    def find_endpoints(self, node, node_list):
        ''' return a endpoints where all nodes have same op and tensor_name
        Args:
            node: dict, the node with allreduce op
            node_list: list, the input nodelist
        '''
        endpoints = []
        for node_itr in node_list:
            if(node_itr['op'] == self.get_plan_type() and
               node_itr['tensor_name'] == node['tensor_name']):
                endpoints.append(node_itr)
        return endpoints

    def separate_allreduce_node(self, node, endpoint):
        '''
        Separating allreduce node includes three step:
        1. generate new primitives nodes of Reduce_Broadcast allreduce
        2. insert primitives nodes into plan
        3. remove original allreduce node from plan

        Args:
            node: dict, the node with allreduce op
            endpoint: list, all node enrolled in the same allreduce operator
        '''
        # all generated nodes are inserted into the index of orignal
        # allreduce node in order
        node_index = self._get_node_index(node)

        # numElements of gradients
        numElements = 1
        for shape in node['output_shapes'][0]:
            numElements *= shape

        # get node device
        device = node['device']

        # get myRank and nRanks
        nRanks = len(endpoint)
        myRank = endpoint.index(node)

        # set rank 0 node as root
        root_node = endpoint[0]
        is_root_node = True if node == root_node else False

        # root gpu receives gradients from all non-root gpus for reducing,
        # then broadcasts reduced gradient to all non-root gpus
        if is_root_node is True:

            # the generated nodes id
            prim_index = 0

            # reduce: root gpu receives gradients from all non-root gpus
            for i in range(1, nRanks):
                non_root_node = endpoint[i]
                target = non_root_node['device']
                self._generate_re_recv_node(node,
                                            prim_index,
                                            node_index,
                                            myRank,
                                            nRanks,
                                            target,
                                            numElements,
                                            device)
                prim_index += 1

            # boradcast: root gpu sends gradients to all non-root gpus
            for i in range(1, nRanks):
                non_root_node = endpoint[i]
                target = non_root_node['device']
                self._generate_bc_send_node(node,
                                            prim_index,
                                            node_index,
                                            myRank,
                                            nRanks,
                                            target,
                                            numElements,
                                            device)
                prim_index += 1

        # non-root gpus send orginal gradients to root gpus for reducing,
        # then receive reduced gradient from root gpu
        else:

            target = root_node['device']

            # reduce: non-root gpu sends gradient to the root gpu
            self._generate_re_send_node(node,
                                        0,
                                        node_index,
                                        myRank,
                                        nRanks,
                                        target,
                                        numElements,
                                        device)

            # boradcast: non-root gpu receives gradients from the root gpu
            self._generate_bc_recv_node(node,
                                        1,
                                        node_index,
                                        myRank,
                                        nRanks,
                                        target,
                                        numElements,
                                        device)

        self._remove_node(node)

    def _generate_re_recv_node(self,
                               node,
                               index,
                               node_index,
                               myRank,
                               nRanks,
                               recvTarget,
                               nelem,
                               device):
        ''' This function generates the recv node of reduce process
        '''
        node_name = node['name'] + '_Recv_' + str(index)
        related_op = node['name'] + '_Send_' + str(0)
        re_recv_node = {'name': node_name,
                        'offset': 0,
                        'size': nelem,
                        'op': 'Recv',
                        'reduction': 'sum',
                        'target': recvTarget,
                        'related_op': related_op,
                        'device': device,
                        'output_shapes': node['output_shapes'],
                        'tensor_name': node['tensor_name'],
                        'tensor_type': node['tensor_type'],
                        'parent': node['name'],
                        'input': node['input'].copy()}

        # The re_recv_node has own dependency on the previous node
        if index != 0:
            pre_node = self._get_node(node_index + index - 1)
            pre_name = pre_node['name']
            re_recv_node['input'].append(pre_name)
        self._add_node(re_recv_node, node_index + index)

    def _generate_re_send_node(self,
                               node,
                               index,
                               node_index,
                               myRank,
                               nRanks,
                               sendTarget,
                               nelem,
                               device):
        ''' This function generates the send node of reduce process
        '''
        node_name = node['name'] + '_Send_' + str(index)
        related_op = node['name'] + '_Recv_' + str(myRank - 1)
        re_send_node = {'name': node_name,
                        'offset': 0,
                        'size': nelem,
                        'op': 'Send',
                        'reduction': '',
                        'target': sendTarget,
                        'related_op': related_op,
                        'device': device,
                        'output_shapes': node['output_shapes'],
                        'tensor_name': node['tensor_name'],
                        'tensor_type': node['tensor_type'],
                        'parent': node['name'],
                        'input': node['input'].copy()}

        # The re_send_node index should always be zero, no dependency
        self._add_node(re_send_node, node_index + index)

    def _generate_bc_send_node(self,
                               node,
                               index,
                               node_index,
                               myRank,
                               nRanks,
                               sendTarget,
                               nelem,
                               device):
        ''' This function generates the send node of broadcast process
        '''
        node_name = node['name'] + '_Send_' + str(index)
        related_op = node['name'] + '_Recv_' + str(1)
        bc_send_node = {'name': node_name,
                        'offset': 0,
                        'size': nelem,
                        'op': 'Send',
                        'reduction': '',
                        'target': sendTarget,
                        'related_op': related_op,
                        'device': device,
                        'output_shapes': node['output_shapes'],
                        'tensor_name': node['tensor_name'],
                        'tensor_type': node['tensor_type'],
                        'parent': node['name'],
                        'input': node['input'].copy()}

        # The bc_send_node has own dependency on the previous node
        if index != 0:
            pre_node = self._get_node(node_index + index - 1)
            pre_name = pre_node['name']
            bc_send_node['input'].append(pre_name)
        self._add_node(bc_send_node, node_index + index)

    def _generate_bc_recv_node(self,
                               node,
                               index,
                               node_index,
                               myRank,
                               nRanks,
                               recvTarget,
                               nelem,
                               device):
        ''' This function generates the recv node of broadcast process
        '''
        node_name = node['name'] + '_Recv_' + str(index)
        related_op = node['name'] + '_Send_' + str(myRank - 1 + nRanks - 1)
        bc_recv_node = {'name': node_name,
                        'offset': 0,
                        'size': nelem,
                        'op': 'Recv',
                        'reduction': 'copy',
                        'target': recvTarget,
                        'related_op': related_op,
                        'device': device,
                        'output_shapes': node['output_shapes'],
                        'tensor_name': node['tensor_name'],
                        'tensor_type': node['tensor_type'],
                        'parent': node['name'],
                        'input': node['input'].copy()}

        # The bc_recv_node has own dependency on the previous send node
        pre = node['name'] + '_Send_' + str(0)
        bc_recv_node['input'].append(pre)
        self._add_node(bc_recv_node, node_index + index)
