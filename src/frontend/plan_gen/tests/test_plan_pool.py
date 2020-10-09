from plan import plan_pool
from plan import plan


def test_plan_pool():

    PlanPool = plan_pool.PlanPool()
    Default_plan = plan.Plan('Default_Plan')
    Test_plan = plan.Plan('Test_Plan')
    plan_name = Default_plan.get_plan_name()
    plan_type = Default_plan.get_plan_type()

    # add our test Plan
    PlanPool.add_plan(Default_plan)

    # Test has_plan()
    assert(PlanPool.has_plan(plan_type, plan_name) is True)
    assert(PlanPool.has_plan("test", "test") is False)
    assert(PlanPool.has_plan(Default_plan) is True)
    assert(PlanPool.has_plan(Test_plan) is False)

    # Test get_plan()
    assert(PlanPool.get_plan(plan_type, plan_name) is Default_plan)
    assert(PlanPool.get_plan("test", "test") is None)

    # Test get_plan_list()
    assert(PlanPool.get_plan_list(plan_type=plan_type) == [Default_plan])
    assert(PlanPool.get_plan_list(plan_type="test") == [])

    # Test delete_plan()
    assert(PlanPool.delete_plan(Default_plan) is True)
    assert(PlanPool.delete_plan(Default_plan) is False)

    # Test reset()
    PlanPool.add_plan(Default_plan)
    PlanPool.add_plan(Test_plan)
    PlanPool.reset()
    assert(PlanPool.has_plan(plan_type, plan_name) is False)
    assert(PlanPool.has_plan(Default_plan) is False)
    assert(PlanPool.get_plan(plan_type, plan_name) is None)
    assert(PlanPool.get_plan_list(plan_type=plan_type) == [])
