import os
import json
from plan import plan_mapper
from resources import resource_pool


def test_gpu_round_robin():
    # Init mapper
    resource_yaml_path = os.path.join(
        os.path.dirname(__file__), 'data', 'resource_pool.yaml')
    rp = resource_pool.ResourcePool()
    rp.init_from_yaml(resource_yaml_path)
    mapper = plan_mapper.GPURoundRobinMapper(rp)

    # Test map function
    path_input = os.path.join(os.path.dirname(__file__),
                              "data/test_generated_plan.json")
    plan = json.load(open(path_input, 'r'))

    mapped_plan = mapper.map(plan)
    path_output = os.path.join(os.path.dirname(__file__),
                               "data/test_mapped_plan.json")

    mappeded_plan_ref = json.load(open(path_output, 'r'))
    assert(mapped_plan == mappeded_plan_ref)

    # None input
    mapped_plan = mapper.map(None)
    assert(mapped_plan is None)

    # Wrong input, assign 5 device into 4 GPUs resource_pool
    plan = [
        {
            "device": "device_0",
            "name": "test",
            "op": "Allreduce",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "input": []
        },
        {
            "device": "device_1",
            "name": "test",
            "op": "Allreduce",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "input": []
        },
        {
            "device": "device_2",
            "name": "test",
            "op": "Allreduce",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "input": []
        },
        {
            "device": "device_3",
            "name": "test",
            "op": "Allreduce",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "input": []
        },
        {
            "device": "device_4",
            "name": "test",
            "op": "Allreduce",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "input": []
        },
    ]
    mapped_plan = mapper.map(plan)
    assert(mapped_plan is None)
