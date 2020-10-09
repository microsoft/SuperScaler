from .plan_mapper import PlanMapper
from .plan_pool import PlanPool


class PlanManager(object):
    """ An manager class controls other plan-related classes including
        plan_pool and plan_mapper
    """
    def __init__(self, plan_pool, plan_mapper):

        if not isinstance(plan_pool, PlanPool):
            raise ValueError("Input plan_pool must be PlanPool instance")
        if not isinstance(plan_mapper, PlanMapper):
            raise ValueError("Input plan_mapper must be PlanMapper instance")

        self.__plan_pool = plan_pool
        self.__plan_mapper = plan_mapper

    def get_execution_plan(self, node_list, plan_type, plan_name):
        ''' Get the execution plan after generation and mapping
        Args:
            node_list: list
            plan_type: string, e.g. Default
            plan_name: string, e.g. Default_Plan
        '''
        plan = self.__plan_pool.get_plan(plan_type=plan_type,
                                         plan_name=plan_name)
        # If the plan_type or plan_name is not correct, plan is None.
        # If the node_list is not a list, we can only get a empty list result
        # so in these cases, we return None as warning
        if plan is None or not isinstance(node_list, list):
            return None
        plan.reset_node_list(node_list)
        output_plan = plan.generate_plan()
        # None output of generate_plan function means something goes wrong
        if output_plan is None:
            return None

        mapped_plan = self.__plan_mapper.map(output_plan)
        # None output of map function means something goes wrong
        if mapped_plan is None:
            return None

        return mapped_plan
