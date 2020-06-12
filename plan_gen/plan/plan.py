import copy


class Plan(object):
    """ An class that generate optimized plan from nodelist.
        target for nodes whose ops equal to plan_type
    """

    def __init__(self,
                 plan_name='Default_Plan',
                 plan_type='Default'):
        ''' Init a plan with name and type
        Args:
            name: string, e.g. Default_Plan
            type: string, e.g. Default
        '''
        self.__plan_name = plan_name
        self.__plan_type = plan_type

        # The major compoment of the plan class
        self.__plan = []

    def get_plan_type(self):
        ''' Get the plan_type attr
        '''
        return self.__plan_type

    def get_plan_name(self):
        ''' Get the plan_name attr
        '''
        return self.__plan_name

    def get_plan_info(self):
        ''' Get the plan_type and plan_name attrs
        '''
        return self.get_plan_type(), self.get_plan_name()

    def reset_plan(self, node_list):
        ''' Set self.__plan from a node_list
        Args:
            node_list: list
        '''
        if not isinstance(node_list, list):
            self.__plan = None
        else:
            self.__plan = copy.deepcopy(node_list)

    def generate_plan(self):
        ''' Main function, will be overloadding by child class
        '''
        return self._get_plan()

    def _add_node(self, node, index=None):
        ''' Add new node into plan.
            If the index is correct, insert node to where index is.
            Otherwise, append node to the end of plan
        Args:
            node: dict
            index: int or None
        '''
        if isinstance(index, int) and index >= 0 and index < len(self.__plan):
            self.__plan.insert(index, node)
        else:
            self.__plan.append(node)

    def _remove_node(self, node):
        ''' Remove node from plan
        Args:
            node: dict
        '''
        if node in self.__plan:
            self.__plan.remove(node)
        else:
            return None

    def _get_plan(self):
        ''' Get plan
        '''
        return self.__plan

    def _get_node_index(self, node):
        ''' Get the index of a node
        Args:
            node: dict
        '''
        if node in self.__plan:
            return self.__plan.index(node)
        else:
            return None

    def _get_node(self, index):
        ''' Get node by index
            If the index is correct, return the node of given index.
            Otherwise, return None as a warning.
        Args:
            index: int
        '''
        if isinstance(index, int) and index >= 0 and index < len(self.__plan):
            return self.__plan[index]
        else:
            return None


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
