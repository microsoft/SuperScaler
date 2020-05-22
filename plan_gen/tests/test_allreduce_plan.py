import copy
from plan import allreduce_plan


def test_plan():

    Allreduce_plan = allreduce_plan.AllreducePlan(plan_name='Allreduce_Plan')
    # Test get_plan_name() function
    assert(Allreduce_plan.get_plan_name() == 'Allreduce_Plan')
    # Test get_plan_type() function
    assert(Allreduce_plan.get_plan_type() == 'Allreduce')
    # Test generate_plan() function
    assert(hasattr(Allreduce_plan, 'generate_plan'))

    nodes = [
        {
         'device': 'device_0',
         'name': 'test',
         'op': 'Allreduce',
         'output_shapes': [[1, 100]],
         'tensor_name': 'test',
         'tensor_type': 1,
         'input': []
        },
        {
         'device': 'device_1',
         'name': 'test',
         'op': 'Allreduce',
         'output_shapes': [[1, 100]],
         'tensor_name': 'test',
         'tensor_type': 1,
         'input': []
        }
        ]

    node = {
            'device': 'device_0',
            'output_shapes': [[1, 100]],
            'name': 'test',
            'op': 'Allreduce',
            'tensor_name': 'test',
            'tensor_type': 1,
            'input': []
            }

    # Test get_num_elements() function
    assert(Allreduce_plan.get_num_elements(node) == 100)

    # Test check_endpoints() function
    endpoints, device = Allreduce_plan.check_endpoints(node, nodes)
    assert(device == 'device_0')
    assert(endpoints == ['device_0', 'device_1'])

    # Test check_endpoints() function
    endpoints, device = Allreduce_plan.check_endpoints(node, nodes)
    assert(device == 'device_0')
    assert(endpoints == ['device_0', 'device_1'])

    # Test append_primitive_list() function
    primitive_list = []
    Allreduce_plan.append_primitive_list(primitive_list)
    assert(primitive_list == [{'endpoints': [],
                               'op': '',
                               'reduction': '',
                               'offset': 0,
                               'size': 0,
                               'index': 0}])

    # Test generate_plan() function
    ''' As the Allreduce_plan is a vitrual class,
        the output of generated Allreduce_plan is
        equal to original
    '''
    node_ref = copy.deepcopy(nodes)
    Allreduce_plan.generate_plan(nodes)
    assert(nodes == node_ref)
