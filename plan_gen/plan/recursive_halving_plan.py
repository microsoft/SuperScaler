import math
from plan.allreduce_plan import AllreducePlan


class RecursiveHalvingAllreducePlan(AllreducePlan):

    def __init__(self, plan_name="Recursive_Halving_Allreduce_Plan"):
        super().__init__(plan_name=plan_name)

    def separate_allreduce_node(self, node, endpoint):
        '''
        Separating allreduce node includes three step:
        1. generate new primitives nodes of Recursive_Halving allreduce
        2. insert primitives nodes into plan
        3. remove original allreduce node from plan

        Args:
            node: dict, the node with allreduce op
            endpoint: list, all node enrolled in the same allreduce operator
        '''
        # all generated nodes are inserted into the location of node_index
        # node_index increase by 1 when new node is generated
        node_index = self._get_node_index(node)

        # numElements of gradients
        numElements = 1
        for shape in node['output_shapes'][0]:
            numElements *= shape

        # get myRank and nRanks
        nRanks = len(endpoint)
        myRank = endpoint.index(node)

        # Recursive_Halving algorithm only achieve allreduce operation when
        # the number of rank (nRank) is exactly the a power of two.
        # Therefore, If nRanks is not a power of two, the allreduceRanks
        # is modified to the largest power of two less than nRanks
        step_size = math.floor(math.log(nRanks, 2))
        allreduceRanks = 2 ** step_size

        # If nRanks is not a power of two, we get r = nRanks - allreduceRanks
        # the first r gpus and the last r gpus are used to do
        # Reduce_broadcast communication before and after recursive_halving

        # input_name is the input_dependency_name for each node
        # input_name is initialized as None for the first generated node
        input_name = None

        # Handling the last r gpus
        if myRank >= allreduceRanks:

            # Create a send node for reduce operator
            target = endpoint[myRank - allreduceRanks]['device']
            node_name = node['name'] + '_reduce_send'
            related_name = node['name'] + '_reduce_recv'
            self._generate_node(node_index=node_index,
                                node_name=node_name,
                                input_name=input_name,
                                related_name=related_name,
                                op='Send',
                                reduction='',
                                offset=0,
                                size=numElements,
                                target=target,
                                node_info=node)
            input_name = node_name
            node_index += 1

            # Create a recv node for broadcast operator
            node_name = node['name'] + '_broadcast_recv'
            related_name = node['name'] + '_broadcast_send'
            self._generate_node(node_index=node_index,
                                node_name=node_name,
                                input_name=input_name,
                                related_name=related_name,
                                op='Recv',
                                reduction='sum',
                                offset=0,
                                size=numElements,
                                target=target,
                                node_info=node)
            input_name = node_name
            node_index += 1

        else:
            # Handling the first r gpus
            if myRank < nRanks - allreduceRanks:
                # Create a recv node for reduce operator
                target = endpoint[myRank - allreduceRanks]['device']
                node_name = node['name'] + '_reduce_recv'
                related_name = node['name'] + '_reduce_send'
                self._generate_node(node_index=node_index,
                                    node_name=node_name,
                                    input_name=input_name,
                                    related_name=related_name,
                                    op='Recv',
                                    reduction='sum',
                                    offset=0,
                                    size=numElements,
                                    target=target,
                                    node_info=node)
                input_name = node_name
                node_index += 1

            # Main components of Recursive_Halving_Allreduce
            # Recursive_Halving_Allreduce is made up of two procedures
            # the Scatter-Reduce and Allgather.

            # The Scatter-Reduce for Recursive and Halving algorithm
            # runs for log(allreduceRanks) steps, where the chunkSize
            # is halved and the distance is doubled

            # Build a proportional sequence as [2,4,...,2^step_size]
            # for the distance of the Scatter-Reduce
            step_list = [2**x for x in range(1, step_size+1)]

            # Scatter-reduce: each gpu sends/recvs gradients to/from its pair
            # the distance of sends/recvs pair is stepRanks/2
            for index, stepRanks in enumerate(step_list):

                stepRank = myRank % stepRanks
                targetRank = (myRank + stepRanks//2) % stepRanks \
                    + myRank // stepRanks * stepRanks

                chunkSizes = [numElements // stepRanks +
                              (i < numElements % stepRanks)
                              for i in range(stepRanks)]
                offsets = [sum(chunkSizes[:i]) for i in range(stepRanks)]

                target = endpoint[targetRank]['device']

                node_name = node['name'] + '_scatter_send_' + str(index)
                related_name = node['name'] + '_scatter_recv_' + str(index)
                self._generate_node(node_index=node_index,
                                    node_name=node_name,
                                    input_name=input_name,
                                    related_name=related_name,
                                    op='Send',
                                    reduction='',
                                    offset=offsets[stepRank],
                                    size=chunkSizes[stepRank],
                                    target=target,
                                    node_info=node)
                input_name = node_name
                node_index += 1

                node_name = node['name'] + '_scatter_recv_' + str(index)
                related_name = node['name'] + '_scatter_send_' + str(index)
                self._generate_node(node_index=node_index,
                                    node_name=node_name,
                                    input_name=input_name,
                                    related_name=related_name,
                                    op='Recv',
                                    reduction='sum',
                                    offset=offsets[targetRank],
                                    size=chunkSizes[targetRank],
                                    target=target,
                                    node_info=node)
                input_name = node_name
                node_index += 1

            # Allgather: each gpu sends/recvs gradients to/from its pair
            # reverse a proportional sequence as [2^step_size,...,4,2]
            # Allgather is the reversed version of scatter-reduce

            # reverse the step_list for Allgather
            step_list.reverse()
            for index, stepRanks in enumerate(step_list):

                stepRank = myRank % stepRanks
                targetRank = (myRank + stepRanks//2) % stepRanks \
                    + myRank // stepRanks * stepRanks

                chunkSizes = [numElements // stepRanks +
                              (i < numElements % stepRanks)
                              for i in range(stepRanks)]
                offsets = [sum(chunkSizes[:i]) for i in range(stepRanks)]

                target = endpoint[targetRank]['device']

                node_name = node['name'] + '_allgather_send_' + str(index)
                related_name = node['name'] + '_allgather_recv_' + str(index)
                self._generate_node(node_index=node_index,
                                    node_name=node_name,
                                    input_name=input_name,
                                    related_name=related_name,
                                    op='Send',
                                    reduction='',
                                    offset=offsets[targetRank],
                                    size=chunkSizes[targetRank],
                                    target=target,
                                    node_info=node)
                input_name = node_name
                node_index += 1

                node_name = node['name'] + '_allgather_recv_' + str(index)
                related_name = node['name'] + '_allgather_send_' + str(index)
                self._generate_node(node_index=node_index,
                                    node_name=node_name,
                                    input_name=input_name,
                                    related_name=related_name,
                                    op='Recv',
                                    reduction='sum',
                                    offset=offsets[stepRank],
                                    size=chunkSizes[stepRank],
                                    target=target,
                                    node_info=node)
                input_name = node_name
                node_index += 1

            # Handling the first r gpus
            if myRank < nRanks - allreduceRanks:
                # broadcast node on the send part
                node_name = node['name'] + '_broadcast_send'
                related_name = node['name'] + '_broadcast_recv'
                self._generate_node(node_index=node_index,
                                    node_name=node_name,
                                    input_name=input_name,
                                    related_name=related_name,
                                    op='Send',
                                    reduction='',
                                    offset=0,
                                    size=numElements,
                                    target=target,
                                    node_info=node)
                input_name = node_name
                node_index += 1

        self._remove_node(node)

    def _generate_node(self,
                       node_index,
                       node_name,
                       input_name,
                       related_name,
                       op,
                       reduction,
                       offset,
                       size,
                       target,
                       node_info):
        ''' generate a node and insert it into plan
        Args:
            node_index: <int> insert generated to the index of plan
            node_name: <str> the name of generated node
            input_name: <str>/<None> the additional input dependency
            related_name: <str> the related op name
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
                          'related_op': related_name,
                          'device': node_info['device'],
                          'output_shapes': node_info['output_shapes'],
                          'tensor_name': node_info['tensor_name'],
                          'tensor_type': node_info['tensor_type'],
                          'parent': node_info['name'],
                          'input': node_info['input'].copy()}

        if input_name is not None:
            generated_node['input'].append(input_name)

        self._add_node(generated_node, node_index)

        return generated_node
