import itertools
from frontend.scaler_graph.IR.node import CompositeNode


def get_output_nodes(graph):
    upstream_ops = set(
        itertools.chain.from_iterable(
            map(
                lambda node: get_input_nodes(node),
                graph.nodes,
            )))
    output_nodes = set(graph.nodes) - upstream_ops
    return output_nodes


def reverse_DFS(graph):
    output_nodes = get_output_nodes(graph)
    temp_nodes = set()
    ordered_nodes = []

    def visit(current_node):
        if current_node in ordered_nodes:
            return
        elif current_node in temp_nodes:
            raise Exception("there is a cycle in graph: %s" %
                            (current_node.name))
        else:
            temp_nodes.add(current_node)
            for input_node in get_input_nodes(current_node):
                visit(input_node)
            temp_nodes.remove(current_node)
            ordered_nodes.append(current_node)

    for current_node in output_nodes:
        visit(current_node)

    return ordered_nodes


def get_input_nodes(node):
    if isinstance(node, CompositeNode):
        raise Exception("We cann't support CompositeNode now.")
    upstream_nodes = set()
    for edge in node.in_edges:
        upstream_nodes.add(edge.src_node)
    return upstream_nodes
