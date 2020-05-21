from plan import plan


def test_plan():

    Default_plan = plan.Plan('Default_Plan')
    # Test get_plan_name() function
    assert(Default_plan.get_plan_name() == 'Default_Plan')
    # Test get_plan_type() function
    assert(Default_plan.get_plan_type() == 'Default')
    # Test generate_plan() function
    assert(hasattr(Default_plan, 'generate_plan'))
