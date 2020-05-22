from plan.plan import Plan


class AllreducePlan(Plan):

    def __init__(self, plan_name="Allreduce_Plan"):
        # The type of plan
        self.plan_type = "Allreduce"
        # The name of plan
        self.plan_name = plan_name
        super().__init__(self.plan_type, self.plan_name)

    @staticmethod
    def __check_node_attr(node):
        ''' Check where the input node follow a correct format.
        '''
        if 'device' not in node:
            raise ValueError("Missing device attr from node: %s" %
                             str(node))

        if 'name' not in node:
            raise ValueError("Missing name attr from node: %s" %
                             str(node))

        if 'op' not in node:
            raise ValueError("Missing op attr from node: %s" %
                             str(node))

        if 'output_shapes' not in node:
            raise ValueError("Missing output_shapes attr from node: %s" %
                             str(node))

        if 'tensor_name' not in node:
            raise ValueError("Missing tensor_name attr from node: %s" %
                             str(node))

        if 'tensor_type' not in node:
            raise ValueError("Missing tensor_type attr from node: %s" %
                             str(node))

        if 'input' not in node:
            raise ValueError("Missing input attr from node: %s" %
                             str(node))

    def generate_plan(self, plan):
        '''
        Generating plan include two step:
        1. split allreduce node into sub primitives list using send/recv or\
            other primitives, then save them into a dict
        2. convert sub primitives list into nodes, and remove the original\
            allreduce node
        '''

        node_book = {}
        # step 1: split allreduce node into send/recv primitives
        for node in plan:
            if 'op' not in node:
                raise ValueError("Missing op attr from node: %s" %
                                 str(node))
            if node['op'] == 'Allreduce':
                # check in advance
                self.__check_node_attr(node)
                '''
                endpoints: the devices enrolled in allreduce primitive
                device: the device of a node
                num_elements: the size of tensor for allreduce
                '''
                endpoints, device = self.check_endpoints(node, plan)
                num_elements = self.get_num_elements(node)
                primitive_list = self.generate_primitive_list(device,
                                                              endpoints,
                                                              num_elements)
                node_book[node['name']] = primitive_list

        # step 2: convert sub primitives into nodes
        for node in plan:
            if node['op'] == 'Allreduce':
                self.insert_nodes(node_book[node['name']], node, plan)

    def generate_primitive_list(self, device, endpoints, num_elements):
        ''' An empty function, will be overridden by inherited classes
        '''
        return []

    def get_num_elements(self, node):
        '''
        get num of elements from the output_shapes attr
        return num_elements
        '''
        # the allreduce primitive should have only one output tensor
        assert(len(node['output_shapes']) == 1)
        num_elements = 1
        for shape in node['output_shapes'][0]:
            num_elements *= shape
        return num_elements

    def check_endpoints(self, node, plan):
        '''
        get device and endpoints from the device attr and the tensor_name attr
        return endpoints, device
        '''
        tensor_name = node['tensor_name']
        endpoints = []
        device = node['device']
        for node_itr in plan:
            if node_itr['op'] == 'Allreduce':
                if node_itr['tensor_name'] == tensor_name:
                    endpoints.append(node_itr['device'])
        return endpoints, device

    def insert_nodes(self, primitive_list, node_ref, plan):
        '''
        convert primitives into nodes, and extend nodelist
        '''
        index = plan.index(node_ref)
        ends = []

        if primitive_list == []:
            # no new primitives are generated, keep unchange
            return
        else:
            for primitive in primitive_list:
                # related_op points the relations between Send/Recv \
                # primitives on different devices
                primitive['name'] = node_ref['name'] + '_' + primitive['op']
                primitive['name'] += '_' + str(primitive['index'])

                if primitive['op'] == 'Send':
                    related_op = node_ref['name'] + '_' + 'Recv'
                    related_op += '_' + str(primitive['related_op'])
                else:
                    related_op = node_ref['name'] + '_' + 'Send'
                    related_op += '_' + str(primitive['related_op'])

                # Create new node
                node_send_recv = {'name': primitive['name'],
                                  'offset': primitive['offset'],
                                  'size': primitive['size'],
                                  'op': primitive['op'],
                                  'reduction': primitive['reduction'],
                                  'target': primitive['endpoints'][0],
                                  'device': node_ref['device'],
                                  'output_shapes': node_ref['output_shapes'],
                                  'tensor_name': node_ref['tensor_name'],
                                  'tensor_type': node_ref['tensor_type'],
                                  'related_op': related_op}

                # Generate input_ids/successor_ids for new node
                if primitive['input_ids']:
                    # non-empty list
                    node_send_recv['input'] = []
                    for input_id in primitive['input_ids']:
                        input_name = primitive_list[input_id]['name']
                        node_send_recv['input'].append(input_name)
                else:
                    # non input
                    node_send_recv['input'] = node_ref['input']

                if len(primitive['successor_ids']) == 0:
                    # empty list
                    ends.append(primitive['name'])

                plan.insert(index, node_send_recv)
                index += 1

            # replace the dependency of allreduce node to Send/Recv nodes
            for node in plan:
                if node_ref['name'] in node['input']:
                    node['input'].remove(node_ref['name'])
                    for end_name in ends:
                        node['input'].append(end_name)

            # remove the old allreduce node
            plan.remove(node_ref)

    def append_primitive_list(self,
                              primitive_list,
                              endpoints=[],
                              op='',
                              reduction='',
                              offset=0,
                              size=0):
        # create Send/Recv primitive
        '''
        primitive def:
            endpoints <list><string>
            op <string>
            reduction <string>
            offset <int>
            size <int>
            index <int>
        '''
        primitive = {'endpoints': endpoints,
                     'op': op,
                     'reduction': reduction,
                     'offset': offset,
                     'size': size,
                     'index': len(primitive_list)}

        primitive_list.append(primitive)
