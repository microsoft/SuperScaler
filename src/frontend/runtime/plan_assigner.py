import copy


class PlanAssigner(object):
    """ An class that assign the communication plan with its corresponding ip.
    """

    def __init__(self):
        super().__init__()

    def assign(self, communication_plan, deployment_setting):
        """ A function that assign the communication_plan
            by the given deployment_setting.

            Returns None if deployment is not legal
            Otherwise, returns assigned_communication_plan

        Args:
          communication_plan: communication plan description provided by
            the plan generator.
          deployment_setting: dict specifying the mapping of hostname and ip.
        """

        # if deployment_setting is illegal, return None
        if not isinstance(deployment_setting, dict):
            return None

        # if communication_plan is illegal, return None
        if not isinstance(communication_plan, list):
            return None

        assigned_communication_plan = copy.deepcopy(communication_plan)
        for plan in assigned_communication_plan:

            # the communication_plan is illegal without host_id
            if 'host_id' not in plan:
                return None

            # the communication_plan should have its corresponding IP address
            if plan['host_id'] not in deployment_setting:
                return None

            plan['ip'] = deployment_setting[plan['host_id']]

        return assigned_communication_plan
