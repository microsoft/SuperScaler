from superscaler.plan_gen.plan.allreduce_plan import AllreducePlan


class RingAllreducePlan(AllreducePlan):

    def __init__(self, plan_name="Ring_Allreduce_Plan"):
        super().__init__(plan_name=plan_name)

    def separate_allreduce_node(self, node, endpoint):
        '''
        Separating allreduce node includes three step:
        1. Generate new primitives nodes of ring allreduce
        2. Insert primitives nodes into node list
        3. Remove the original allreduce node from node list

        Args:
            node: dict, the node with allreduce op
            endpoint: list, all node enrolled in the same allreduce operator
        '''
        # All generated nodes are inserted into the index of orignal
        # allreduce node in order
        node_index = self._get_node_list().index(node)

        # numElements of gradients for ring_allreduce
        numElements = 1
        for shape in node.output_shapes[0]:
            numElements *= shape

        # Get myRank and nRanks in ring allreduce
        nRanks = len(endpoint)
        myRank = endpoint.index(node)

        # input_name is the input_dependency_name for each node
        # input_name is initialized as None for the first generated node
        # When a new node generated, the input_name is assigned by node_name
        input_name = None

        # sendTarget indicates the gpu of the next rank node
        # recvTarget indicates the gpu of the previous rank node
        sendTarget = endpoint.get_node((myRank + 1) % nRanks).device
        recvTarget = endpoint.get_node((myRank + nRanks - 1) % nRanks).device

        # chunkSizes illustrates the data size for each send/recv
        # offsets illustrates the data address for each send/recv
        # sendIndex and receiveIndex illustares the offset
        # note that num_elements can not divided by count, each
        # send/recv will transfer different data
        chunkSizes = [numElements // nRanks + (i < numElements % nRanks)
                      for i in range(nRanks)]
        offsets = [sum(chunkSizes[:i]) for i in range(nRanks)]
        sendIndex = myRank
        receiveIndex = (myRank + nRanks - 1) % nRanks

        # Scatter-reduce: each gpu sends gradients to the next gpu,
        # and receives gradients from the previous gpu in ring.
        # Finally each GPU will contain a part of reduced gradients
        for index in range(nRanks - 1):
            # Generate send node for scatter-reduce

            node_name = node.name + '_scatter_send_' + str(index)
            target_name = node.name + '_scatter_recv_' + str(index)
            self._generate_node(node_index=node_index,
                                node_name=node_name,
                                input_name=input_name,
                                target_name=target_name,
                                op='Send',
                                reduction='',
                                offset=offsets[sendIndex],
                                size=chunkSizes[sendIndex],
                                target=sendTarget,
                                node_info=node)
            input_name = node_name
            node_index += 1
            sendIndex = (sendIndex + nRanks - 1) % nRanks

            # Generate recv node for scatter-reduce
            node_name = node.name + '_scatter_recv_' + str(index)
            target_name = node.name + '_scatter_send_' + str(index)
            self._generate_node(node_index=node_index,
                                node_name=node_name,
                                input_name=input_name,
                                target_name=target_name,
                                op='Recv',
                                reduction='sum',
                                offset=offsets[receiveIndex],
                                size=chunkSizes[receiveIndex],
                                target=recvTarget,
                                node_info=node)
            input_name = node_name
            node_index += 1
            receiveIndex = (receiveIndex + nRanks - 1) % nRanks

        # Allgather: GPUs will gather gradients from ring.
        # Finally all GPUs will get reduced gradients
        for index in range(nRanks - 1):
            # Generate send node for allgather
            node_name = node.name + '_allgather_send_' + str(index)
            target_name = node.name + '_allgather_recv_' + str(index)
            self._generate_node(node_index=node_index,
                                node_name=node_name,
                                input_name=input_name,
                                target_name=target_name,
                                op='Send',
                                reduction='',
                                offset=offsets[sendIndex],
                                size=chunkSizes[sendIndex],
                                target=sendTarget,
                                node_info=node)
            input_name = node_name
            node_index += 1
            sendIndex = (sendIndex + nRanks - 1) % nRanks

            # Generate recv node for allgather
            node_name = node.name + '_allgather_recv_' + str(index)
            target_name = node.name + '_allgather_send_' + str(index)
            self._generate_node(node_index=node_index,
                                node_name=node_name,
                                input_name=input_name,
                                target_name=target_name,
                                op='Recv',
                                reduction='copy',
                                offset=offsets[receiveIndex],
                                size=chunkSizes[receiveIndex],
                                target=recvTarget,
                                node_info=node)
            input_name = node_name
            node_index += 1
            receiveIndex = (receiveIndex + nRanks - 1) % nRanks
        self._get_node_list().remove(node)
