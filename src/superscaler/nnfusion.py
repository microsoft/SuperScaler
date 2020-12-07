import os
import json
from superscaler.superscaler import SuperscalerError
from superscaler.plan_gen.plan.parser.tf_parser import TFParser
from superscaler.plan_gen.plan.resources.resource_pool import ResourcePool
from superscaler.plan_gen.plan.plan_generator import PlanGenerator
from superscaler.plan_gen.plan.adapter.superscaler_adapter import \
     SuperScalerAdapter


def generate_data_parallelism_plan(protobuf_path,
                                   parallelism_level,
                                   resource_pool,
                                   output_dir,
                                   communication_DSL='ring'):
    """ A function that generates communication_plan for NNfusion graph.

    Args:
        protobuf_path: string represenation of NNfusion graph
        parallelism_level: int specifying number of parallelism_level
        resource_pool: JSON file specifying hardware description and network
        topology.
        output_dir: str specifying which path to output tmp files
        communication_DSL: domain specific language to decribe communication
    """
    if not isinstance(resource_pool, str):
        raise SuperscalerError("resource_pool should be str")
    if not isinstance(parallelism_level, int):
        raise SuperscalerError("parallelism_level should be int")
    if not isinstance(resource_pool, str):
        raise SuperscalerError("resource_pool should be inited from file")
    if not isinstance(output_dir, str):
        raise SuperscalerError("output_dir should be str")
    if not isinstance(communication_DSL, str):
        raise SuperscalerError("communication_DSL should be str")

    def get_device(parallelism_level):
        # get virtual device names
        return ["device_%d" % (i) for i in range(parallelism_level)]

    def get_graph_paths(protobuf_path, parallelism_level):
        # get multiple protobufs
        graph_paths = []
        for i in range(parallelism_level):
            graph_paths.append(protobuf_path)
        return graph_paths

    parser = TFParser()
    devices = get_device(parallelism_level)
    graph_paths = get_graph_paths(protobuf_path, parallelism_level)

    # Parse NNfusion graph as nodelist
    nodelist = parser.parse_graphs(graph_paths, devices)

    # Init plan_generator
    rp = ResourcePool()
    rp.init_from_yaml(resource_pool)

    plan_generator = PlanGenerator(nodelist, rp)

    # Generate communication plan
    plan = plan_generator.get_execution_plan('Allreduce', communication_DSL)

    # Adapte communication plan for SuperScaler
    adapter = SuperScalerAdapter()
    adapter.set_plan(plan)

    output_plan = adapter.adapt_plan()

    # Dump output_plan to output_dir
    for i in range(parallelism_level):
        tmp_rank_dir = os.path.join(output_dir, str(i))
        if not os.path.exists(tmp_rank_dir):
            os.makedirs(tmp_rank_dir)

        plan_path = os.path.join(tmp_rank_dir, 'plan.json')
        json.dump(output_plan[i],
                  open(plan_path, 'w'),
                  indent=4,
                  sort_keys=True)
