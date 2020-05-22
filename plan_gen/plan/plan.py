class Plan(object):

    def __init__(self, plan_type="Default", plan_name="Default_Plan"):
        # The type of plan
        self.plan_type = plan_type
        # The name of plan
        self.plan_name = plan_name

    def generate_plan(self, plan):
        ''' An empty function, will be overridden by inherited classes
        '''
        pass

    def get_plan_type(self):
        ''' get the plan_type attr
        '''
        return self.plan_type

    def get_plan_name(self):
        ''' get the plan_name attr
        '''
        return self.plan_name
