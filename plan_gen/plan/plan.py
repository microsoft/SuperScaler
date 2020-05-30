import copy


class Plan(object):
    """ An class that generate optimized plan from nodelist.
        target for nodes whose op equal to plan_type
    """

    def __init__(self,
                 plan_name='Default_Plan',
                 plan_type='Default'):
        # The name of plan class
        self.__plan_name = plan_name
        # The type of plan class
        self.__plan_type = plan_type
        # The plan used in this class
        self.__plan = []

    def get_plan_type(self):
        ''' Get the plan_type attr
        '''
        return self.__plan_type

    def get_plan_name(self):
        ''' Get the plan_name attr
        '''
        return self.__plan_name

    def reset_plan(self, node_list):
        ''' Set plan vary from a node_list
        '''
        if node_list is None:
            self.__plan = []
        else:
            self.__plan = copy.deepcopy(node_list)

    def generate_plan(self):
        ''' Main function, will be overloadding by child class
        '''
        return self._get_plan()

    def _add_node(self, node, index=None):
        ''' add new node into plan
        '''
        if isinstance(index, int) and index >= 0 and index < len(self.__plan):
            self.__plan.insert(index, node)
        else:
            self.__plan.append(node)

    def _remove_node(self, node):
        ''' remove node from plan
        '''
        if node in self.__plan:
            self.__plan.remove(node)
        else:
            return

    def _get_plan(self):
        ''' get plan
        '''
        return self.__plan

    def _get_node_index(self, node):
        ''' remove node from plan
        '''
        if node in self.__plan:
            return self.__plan.index(node)
        else:
            return None
