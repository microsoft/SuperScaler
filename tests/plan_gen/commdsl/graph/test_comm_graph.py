# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.commdsl.graph.node import Node, CompNode, TransNode
from superscaler.plan_gen.commdsl.graph.meta import TransNodeType
from superscaler.plan_gen.commdsl.graph.meta import CompNodeType
from superscaler.plan_gen.commdsl.graph.graph import CommGraph
from superscaler.plan_gen.commdsl.graph.segment import DataSegment
from superscaler.plan_gen.commdsl.errors import CommDSLRuntimeError
import pytest


def test_graph_property():
    graph = CommGraph()
    assert len(graph.nodes) == 1
    # test visibility
    _ = graph.nodes[0]
    assert graph.adj is None
    with pytest.raises(TypeError):
        graph.nodes = Node()
    with pytest.raises(CommDSLRuntimeError):
        graph.adj = Node()


def test_comm_graph_core():
    seg1 = DataSegment(1)
    seg2 = DataSegment(1)
    seg3 = DataSegment(1)
    graph = CommGraph()
    tnode1 = TransNode(TransNodeType.SEND, sendbuf=seg1, dst=1)
    tnode2 = TransNode(TransNodeType.RECV, recvbuf=seg2, src=2)
    cnode1 = CompNode(CompNodeType.ADD, seg3, seg2, seg3, True)
    tnode3 = TransNode(TransNodeType.SEND, sendbuf=seg2, dst=2)
    # test add_op
    graph.add_op(tnode1)
    graph.add_op(tnode2)
    graph.add_op(cnode1)
    graph.add_op(tnode3)
    assert len(graph.nodes) == 5
    assert tnode1.name == TransNodeType.SEND.value + '_0'
    assert tnode2.name == TransNodeType.RECV.value + '_0'
    assert cnode1.name == CompNodeType.ADD.value + '_0'
    assert tnode3.name == TransNodeType.SEND.value + '_1'
    # test generated graph
    graph.gen_graph()
    assert graph.adj[0, 1] == 1
    assert graph.adj[0, 2] == 1
    assert graph.adj[0, 3] == 0
    assert graph.adj[1, 2] == 0
    assert graph.adj[2, 3] == 1
    # test depend_nodes
    nodes, nids = graph.depend_nodes(cnode1)
    assert len(nodes) == 1 and len(nids) == 1
    assert nodes[0] == tnode2
    assert graph.nodes.index(tnode2) == nids[0]
    nodes, nids = graph.depend_nodes(tnode2)
    assert len(nodes) == 1 and len(nids) == 1
    # test succ_nodes
    nodes, nids = graph.succ_nodes(tnode2)
    assert len(nodes) == 2 and len(nids) == 2
    assert nodes[0] == cnode1 and nodes[1] == tnode3
    nodes, nids = graph.succ_nodes(tnode1)
    assert len(nodes) == 0 and len(nids) == 0
    # test remove_op
    graph.remove_op(tnode2)
    assert len(graph.nodes) == 4
    nodes, nids = graph.depend_nodes(cnode1)
    assert len(nodes) == 0 and len(nids) == 0


def test_comm_graph_get_node():
    seg1 = DataSegment(1)
    seg2 = DataSegment(1)
    seg3 = DataSegment(1)
    graph = CommGraph()
    tnode1 = TransNode(TransNodeType.SEND, sendbuf=seg1, dst=1)
    tnode2 = TransNode(TransNodeType.RECV, recvbuf=seg2, src=2)
    cnode1 = CompNode(CompNodeType.ADD, seg3, seg2, seg3, True)
    graph.add_op(tnode1)
    graph.add_op(tnode2)
    graph.add_op(cnode1)
    # test get_node
    node = graph.get_node(1, conds=dict(r_segs=[seg1], dst=1))
    assert node == tnode1
    node = graph.get_node(1)
    assert node == tnode1
    node = graph.get_node(1, conds=dict(op=CompNodeType.ADD))
    assert node == cnode1
    node = graph.get_node(2, conds=dict(sendbuf=seg1, dst=1))
    assert node is None
    node = graph.get_node(1, conds=dict(dst=2))
    assert node is None
    node = graph.get_node(1, conds=dict(reduction=CompNodeType.COPY))
    assert node == tnode2
    with pytest.raises(CommDSLRuntimeError):
        graph.get_node(0, conds=dict(sendbuf=seg1, dst=1))


def test_comm_graph_node_count():
    seg1 = DataSegment(1)
    seg2 = DataSegment(1)
    seg3 = DataSegment(1)
    cnode1 = CompNode(CompNodeType.ADD, seg3, seg2, seg3, True)
    cnode2 = CompNode(CompNodeType.MUL, seg3, seg2, seg1, False)
    cnode3 = CompNode(CompNodeType.ADD, seg1, seg2, seg3, False)
    graph = CommGraph()
    graph.add_op(cnode1)
    graph.add_op(cnode2)
    graph.add_op(cnode3)
    # test node count with both parameters
    count = graph.get_node_count(last=cnode1, conds=dict(op=cnode1.op))
    assert count == 1
    count = graph.get_node_count(last=cnode2, conds=dict(lhs=cnode1.lhs))
    assert count == 2
    count = graph.get_node_count(
        last=cnode2, conds=dict(output=cnode1.output))
    assert count == 1
    count = graph.get_node_count(
        last=cnode1, conds=dict(op=TransNodeType.SEND))
    assert count == 0
    # test node count with only conds
    count = graph.get_node_count(conds=dict(op=CompNodeType.ADD))
    assert count == 2
    count = graph.get_node_count(conds=dict(op=CompNodeType.DIV))
    assert count == 0
    # test node count with only last
    count = graph.get_node_count(last=cnode2)
    assert count == 3  # note there will be a start empty node
    out_node = CompNode(CompNodeType.ADD, seg3, seg2, seg3, True)
    with pytest.raises(CommDSLRuntimeError):
        _ = graph.get_node_count(last=out_node, conds=dict(op=cnode1.op))
