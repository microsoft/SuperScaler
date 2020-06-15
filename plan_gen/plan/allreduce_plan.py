import copy
from plan.plan import Plan


class AllreducePlan(Plan):
    """ An class that generates a optimized plan from nodelist.
        Targeting for nodes with Allreduce op.
    """

    def __init__(self, plan_name):
        ''' Init a plan with name and set plan_type as Allreduce internally
        Args:
            name: string, e.g. Allreduce_Plan
            type: string, e.g. Allreduce
        '''
        super().__init__(plan_type="Allreduce",
                         plan_name=plan_name)

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
        Virtual funtion

        Args:
            node: dict, the node with allreduce op
            endpoint: list, all node enrolled in the same allreduce operator
        '''
        return
