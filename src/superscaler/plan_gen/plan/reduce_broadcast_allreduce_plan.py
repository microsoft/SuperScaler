from superscaler.plan_gen.plan.allreduce_plan import AllreducePlan


class ReduceBroadcastAllreducePlan(AllreducePlan):

    def __init__(self, plan_name="Reduce_Broadcast_Allreduce_Plan"):
        super().__init__(plan_name=plan_name)

    def separate_allreduce_node(self, node, endpoint):
        '''
        Separating allreduce node includes three step:
        1. generate new primitives nodes of Reduce_Broadcast allreduce
        2. insert primitives nodes into node list
        3. remove original allreduce node from node list

        Args:
            node: dict, the node with allreduce op
            endpoint: list, all node enrolled in the same allreduce operator
        '''
        # all generated nodes are inserted into the index of orignal
        # allreduce node in order
        node_index = self._get_node_list().index(node)

        # numElements of gradients
        numElements = 1
        for shape in node.output_shapes[0]:
            numElements *= shape

        # get myRank and nRanks
        nRanks = len(endpoint)
        myRank = endpoint.index(node)

        # set rank 0 node as root
        root_node = endpoint.get_node(0)
        is_root_node = True if node == root_node else False

        # input_name is the input_dependency_name for each node
        # input_name is initialized as None for the first generated node
        # When a new node generated, the input_name is assigned by node_name
        input_name = None

        # root gpu receives gradients from all non-root gpus for reducing,
        # then broadcasts reduced gradient to all non-root gpus
        if is_root_node is True:

            # reduce: root gpu receives gradients from all non-root gpus
            for index in range(1, nRanks):
                non_root_node = endpoint.get_node(index)
                target = non_root_node.device
                node_name = node.name + '_reduce_recv' + str(index-1)
                target_name = node.name + '_reduce_send' + str(index-1)
                self._generate_node(node_index=node_index,
                                    node_name=node_name,
                                    input_name=input_name,
                                    target_name=target_name,
                                    op='Recv',
                                    reduction='sum',
                                    offset=0,
                                    size=numElements,
                                    target=target,
                                    node_info=node)
                input_name = node_name
                node_index += 1

            # boradcast: root gpu sends gradients to all non-root gpus
            for index in range(1, nRanks):
                non_root_node = endpoint.get_node(index)
                target = non_root_node.device
                node_name = node.name + '_broadcast_send' + str(index-1)
                target_name = node.name + '_broadcast_recv' + str(index-1)
                self._generate_node(node_index=node_index,
                                    node_name=node_name,
                                    input_name=input_name,
                                    target_name=target_name,
                                    op='Send',
                                    reduction='',
                                    offset=0,
                                    size=numElements,
                                    target=target,
                                    node_info=node)
                input_name = node_name
                node_index += 1

        # non-root gpus send orginal gradients to root gpus for reducing,
        # then receive reduced gradient from root gpu
        else:

            target = root_node.device

            # reduce: non-root gpu sends gradient to the root gpu
            node_name = node.name + '_reduce_send' + str(myRank-1)
            target_name = node.name + '_reduce_recv' + str(myRank-1)
            self._generate_node(node_index=node_index,
                                node_name=node_name,
                                input_name=input_name,
                                target_name=target_name,
                                op='Send',
                                reduction='',
                                offset=0,
                                size=numElements,
                                target=target,
                                node_info=node)

            # boradcast: non-root gpu receives gradients from the root gpu
            node_name = node.name + '_broadcast_recv' + str(myRank-1)
            target_name = node.name + '_broadcast_send' + str(myRank-1)
            self._generate_node(node_index=node_index,
                                node_name=node_name,
                                input_name=input_name,
                                target_name=target_name,
                                op='Recv',
                                reduction='copy',
                                offset=0,
                                size=numElements,
                                target=target,
                                node_info=node)

        self._get_node_list().remove(node)
