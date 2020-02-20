import json
import pickle

import resource
import ssg_visitor
import device_assigner
import operator_assigner
import plan_generator


def test_plan_generator():
    rp = pickle.load(open("tests/data/resource_pool.data", "rb"))
    graphs = json.load(open("tests/data/ssg_graph.json", "r"))
    if isinstance(graphs, list):
        graphs = {i : graphs[i] for i in range(len(graphs))}
    plans_expected = json.load(open("tests/data/plans.json", "r"))
    pg = plan_generator.PlanGenerator(
        graph_visitor = ssg_visitor.JsonVisitor,
        resource_pool = rp
        )
    plans = pg.get_plan(graphs)
    # # Dump plans
    # json.dump(
    #     plans,
    #     open("tests/data/plans.json", "w"),
    #     indent=4,
    #     sort_keys=True)
    assert(plans == plans_expected)
