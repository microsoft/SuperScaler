import json

import ssg_visitor


def test_ssg_visitor():
    graph_json = json.load(open("tests/data/ssg_graph.json", "r"))
    for graph in graph_json:
        for op in ssg_visitor.JsonVisitor(graph):
            assert(op)
            assert(op.name)
            assert(op.tensor)
