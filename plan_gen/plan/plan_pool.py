from plan.plan import Plan


class PlanPool(object):
    def __init__(self):
        self.__plan_pool = {}

    def reset(self):
        ''' reset plan pool to a clean dict
        '''
        self.__plan_pool.clear()

    def has_plan(*args):
        ''' check whether a plan exists

            The has_plan() function is overloadding as two format:
            1, has_plan(self, plan):
                plan: plan class
            2, has_plan(self, plan_type, plan_name)
                plan_type: the type of plan
                plan_name: the name of plan
        '''
        if len(args) == 2 and isinstance(args[1], Plan):
            self = args[0]
            plan = args[1]
            plan_type = plan.get_plan_type()
            plan_name = plan.get_plan_name()
        elif len(args) == 3 and (isinstance(args[1], str)
                                 and isinstance(args[2], str)):
            self = args[0]
            plan_type = args[1]
            plan_name = args[2]
        else:
            raise Exception("Wrong input format")

        if plan_type not in self.__plan_pool:
            return False
        elif plan_name not in self.__plan_pool[plan_type]:
            return False
        else:
            return True

    def get_plan(self, plan_type, plan_name):
        ''' get a plan by the indexs of plan_type and plan_name
        '''
        if self.has_plan(plan_type, plan_name):
            return self.__plan_pool[plan_type][plan_name]
        else:
            return None

    def get_plan_list(self, plan_type):
        ''' get a list of plan with same plan_type
        '''
        if plan_type not in self.__plan_pool:
            return []
        else:
            return [plan for _, plan in self.__plan_pool[plan_type].items()]

    def delete_plan(self, plan):
        ''' delete a plan
        '''
        if self.has_plan(plan):
            plan_type = plan.get_plan_type()
            plan_name = plan.get_plan_name()
            self.__plan_pool[plan_type].pop(plan_name)
            return True
        else:
            return False

    def add_plan(self, plan):
        ''' add a plan into plan pool
        '''
        plan_type = plan.get_plan_type()
        plan_name = plan.get_plan_name()
        if plan_type not in self.__plan_pool:
            self.__plan_pool[plan_type] = {}
        self.__plan_pool[plan_type][plan_name] = plan
