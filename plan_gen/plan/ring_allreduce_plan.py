import copy
from plan.plan import AllreducePlan


class RingAllreducePlan(AllreducePlan):

    def __init__(self, plan_name="Ring_Allreduce_Plan"):
        super().__init__(plan_name=plan_name)

    def generate_plan(self):
        '''
        Generating plan includes three step:
        1. Find all allreudce nodes as allreduce_node_list
        2. For a specific node, find its related nodes as endpoints
        3. Separate all allreduce node to ring allreduce nodes
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
        ''' Return a allreduce_node_list with allreduce op
        Args:
            node_list: list, the input nodelist
        '''
        allreduce_node_list = []
        for node in node_list:
            if node['op'] == self.get_plan_type():
                allreduce_node_list.append(node)
        return allreduce_node_list

    def find_endpoints(self, node, node_list):
        ''' Return a endpoints where all nodes have same op and tensor_name
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
        1. Generate new primitives nodes of ring allreduce
        2. Insert primitives nodes into plan
        3. Remove the original allreduce node from plan

        Args:
            node: dict, the node with allreduce op
            endpoint: list, all node enrolled in the same allreduce operator
        '''
        # All generated nodes are inserted into the index of orignal
        # allreduce node in order
        node_index = self._get_node_index(node)

        # numElements of gradients for ring_allreduce
        numElements = 1
        for shape in node['output_shapes'][0]:
            numElements *= shape

        # Get node device
        device = node['device']

        # Get myRank and nRanks in ring allreduce
        nRanks = len(endpoint)
        myRank = endpoint.index(node)

        # sendTarget indicates the gpu of the next rank node
        # recvTarget indicates the gpu of the previous rank node
        sendTarget = endpoint[(myRank + 1) % nRanks]['device']
        recvTarget = endpoint[(myRank + nRanks - 1) % nRanks]['device']

        # chunkSizes illustrates the data size for each send/recv
        # offsets illustrates the data address for each send/recv
        # sendIndex and receiveIndex illustares the offset
        # note that num_elements can not divided by count, each
        # send/recv will transfer different data
        chunkSizes = [numElements // nRanks + (i < numElements % nRanks)
                      for i in range(nRanks)]
        offsets = [0 for _ in range(nRanks)]
        for i in range(nRanks):
            if i != 0:
                offsets[i] += chunkSizes[i-1]
        sendIndex = myRank
        receiveIndex = (myRank + nRanks - 1) % nRanks

        # The generated nodes id
        prim_index = 0

        # Scatter-reduce: each gpu sends gradients to the next gpu,
        # and receives gradients from the previous gpu in ring.
        # Finally each GPU will contain a part of reduced gradients
        for _ in range(nRanks - 1):
            # Generate sr_send_node
            sendAddress = offsets[sendIndex]
            nelem = chunkSizes[sendIndex]
            sendIndex = (sendIndex + nRanks - 1) % nRanks

            sr_send_node = self._generate_sr_send_node(node,
                                                       prim_index,
                                                       sendAddress,
                                                       sendTarget,
                                                       nelem,
                                                       device)

            self._add_node(sr_send_node, node_index + prim_index)
            prim_index += 1

            # Generate sr_recv_node
            receiveAddress = offsets[receiveIndex]
            nelem = chunkSizes[receiveIndex]
            receiveIndex = (receiveIndex + nRanks - 1) % nRanks

            sr_recv_node = self._generate_sr_recv_node(node,
                                                       prim_index,
                                                       receiveAddress,
                                                       recvTarget,
                                                       nelem,
                                                       device)
            self._add_node(sr_recv_node, node_index + prim_index)
            prim_index += 1

        # Allgather: GPUs will gather gradients from ring.
        # Finally all GPUs will get reduced gradients
        for _ in range(nRanks - 1):
            # Generate ag_send_node
            sendAddress = offsets[sendIndex]
            nelem = chunkSizes[sendIndex]
            sendIndex = (sendIndex + nRanks - 1) % nRanks

            ag_send_node = self._generate_ag_send_node(node,
                                                       prim_index,
                                                       sendAddress,
                                                       sendTarget,
                                                       nelem,
                                                       device)
            self._add_node(ag_send_node, node_index + prim_index)
            prim_index += 1

            # Generate ag_recv_node
            receiveAddress = offsets[receiveIndex]
            nelem = chunkSizes[receiveIndex]
            receiveIndex = (receiveIndex + nRanks - 1) % nRanks

            ag_recv_node = self._generate_ag_recv_node(node,
                                                       prim_index,
                                                       receiveAddress,
                                                       recvTarget,
                                                       nelem,
                                                       device)
            self._add_node(ag_recv_node, node_index + prim_index)
            prim_index += 1
        self._remove_node(node)

    @staticmethod
    def _generate_sr_send_node(node,
                               index,
                               sendAddress,
                               sendTarget,
                               nelem,
                               device):
        ''' This function generates the send node of scatter-reduce process
        '''
        node_name = node['name'] + '_Send_' + str(index)
        related_op = node['name'] + '_Recv_' + str(index + 1)
        sr_send_node = {'name': node_name,
                        'offset': sendAddress,
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

        # The sr_send_node has own dependency on previous recv node
        if index != 0:
            pre = node['name'] + '_Recv_' + str(index - 1)
            sr_send_node['input'].append(pre)
        return sr_send_node

    @staticmethod
    def _generate_sr_recv_node(node,
                               index,
                               receiveAddress,
                               recvTarget,
                               nelem,
                               device):
        ''' This function generates the recv node of scatter-reduce process
        '''
        node_name = node['name'] + '_Recv_' + str(index)
        related_op = node['name'] + '_Send_' + str(index - 1)
        sr_recv_node = {'name': node_name,
                        'offset': receiveAddress,
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

        # The sr_recv_node has own dependency on previous send node
        if index != 0:
            pre = node['name'] + '_Send_' + str(index - 1)
            sr_recv_node['input'].append(pre)
        return sr_recv_node

    @staticmethod
    def _generate_ag_send_node(node,
                               index,
                               sendAddress,
                               sendTarget,
                               nelem,
                               device):
        ''' This function generates the send node of allgather process
        '''
        node_name = node['name'] + '_Send_' + str(index)
        related_op = node['name'] + '_Recv_' + str(index + 1)
        ag_send_node = {'name': node_name,
                        'offset': sendAddress,
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

        # The ag_send_node has own dependency on previous recv node
        if index != 0:
            pre = node['name'] + '_Recv_' + str(index - 1)
            ag_send_node['input'].append(pre)
        return ag_send_node

    @staticmethod
    def _generate_ag_recv_node(node,
                               index,
                               receiveAddress,
                               recvTarget,
                               nelem,
                               device):
        ''' This function generates the recv node of allgather process
        '''
        node_name = node['name'] + '_Recv_' + str(index)
        related_op = node['name'] + '_Send_' + str(index - 1)
        ag_recv_node = {'name': node_name,
                        'offset': receiveAddress,
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

        # The ag_recv_node has own dependency on previous send node
        if index != 0:
            pre = node['name'] + '_Send_' + str(index - 1)
            ag_recv_node['input'].append(pre)
        return ag_recv_node
