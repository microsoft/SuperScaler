# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.plan.plan import Plan


class PlanPool(object):
    ''' A Poll class uses a dict to store different plan classes
    '''
    def __init__(self):
        self.__plan_pool = {}

    def reset(self):
        ''' Reset plan pool to a clean dict
        '''
        self.__plan_pool.clear()

    def has_plan(*args):
        ''' Check whether a plan exists

            The has_plan() function is overloadding as two format:
            1, has_plan(self, plan):
                plan: plan class
            2, has_plan(self, plan_type, plan_name)
                plan_type: the type of plan
                plan_name: the name of plan
        '''
        if len(args) == 2 and isinstance(args[1], Plan):
            self, plan = args
            plan_type, plan_name = plan.get_plan_info()
        elif (len(args) == 3 and isinstance(args[1], str) and
              isinstance(args[2], str)):
            self, plan_type, plan_name = args
        else:
            return False

        if plan_type not in self.__plan_pool:
            return False
        elif plan_name not in self.__plan_pool[plan_type]:
            return False
        else:
            return True

    def get_plan(self, plan_type, plan_name):
        ''' Get a plan by the indexs of plan_type and plan_name
        '''
        if self.has_plan(plan_type, plan_name):
            return self.__plan_pool[plan_type][plan_name]
        else:
            return None

    def get_plan_list(self, plan_type):
        ''' Get a list of plan with same plan_type
        '''
        if plan_type not in self.__plan_pool:
            return []
        else:
            return [plan for _, plan in self.__plan_pool[plan_type].items()]

    def delete_plan(self, plan):
        ''' Delete a plan from the plan pool
        '''
        if self.has_plan(plan):
            plan_type, plan_name = plan.get_plan_info()
            self.__plan_pool[plan_type].pop(plan_name)
            return True
        else:
            return False

    def add_plan(self, plan):
        ''' Add a plan into the plan pool
        '''
        plan_type, plan_name = plan.get_plan_info()
        if plan_type not in self.__plan_pool:
            self.__plan_pool[plan_type] = {}
        self.__plan_pool[plan_type][plan_name] = plan
