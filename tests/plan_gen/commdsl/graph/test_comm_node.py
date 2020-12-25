# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.commdsl.graph.node import Node, CompNode, TransNode
from superscaler.plan_gen.commdsl.graph.meta import TransNodeType
from superscaler.plan_gen.commdsl.graph.meta import CompNodeType
from superscaler.plan_gen.commdsl.graph.segment import DataSegment
from superscaler.plan_gen.commdsl.errors import CommDSLRuntimeError
import pytest


def test_base_node_init():
    bnode = Node()
    assert bnode.name is None
    assert bnode.op is None
    assert len(bnode.r_segs) == 0
    assert len(bnode.w_segs) == 0


def test_node_add_read_segs():
    seg1 = DataSegment(1)
    seg2 = DataSegment(1)
    bnode = Node()
    # test add_read_segs
    bnode.add_read_segs(seg1)
    assert len(bnode.r_segs) == 1 and bnode.r_segs[0] == seg1
    bnode.add_read_segs([seg1, seg2])
    assert len(bnode.r_segs) == 3
    with pytest.raises(TypeError):
        bnode.add_read_segs([seg1, 2])
    with pytest.raises(TypeError):
        bnode.add_read_segs(2)


def test_node_reset_read_segs():
    seg1 = DataSegment(1)
    seg2 = DataSegment(1)
    bnode = Node()
    # test reset read segs
    bnode.add_read_segs([seg1, seg2])
    assert len(bnode.r_segs) == 2
    bnode.reset_read_segs()
    assert len(bnode.r_segs) == 0


def test_node_add_write_segs():
    seg1 = DataSegment(1)
    seg2 = DataSegment(1)
    bnode = Node()
    # test add_write_segs
    bnode.add_write_segs(seg1)
    assert len(bnode.w_segs) == 1 and bnode.w_segs[0] == seg1
    bnode.add_write_segs([seg1, seg2])
    assert len(bnode.w_segs) == 3
    with pytest.raises(TypeError):
        bnode.add_write_segs([seg1, 2])
    with pytest.raises(TypeError):
        bnode.add_write_segs(2)


def test_node_reset_write_segs():
    seg1 = DataSegment(1)
    seg2 = DataSegment(1)
    bnode = Node()
    # test reset read segs
    bnode.add_write_segs([seg1, seg2])
    assert len(bnode.w_segs) == 2
    bnode.reset_write_segs()
    assert len(bnode.w_segs) == 0


def test_node_depend_on():
    seg1 = DataSegment(1)
    bnode_early = Node()
    bnode_later = Node()
    # test read-write dependency
    bnode_early.add_read_segs(seg1)
    bnode_later.add_write_segs(seg1)
    assert bnode_later.depend_on(bnode_early) is True
    # test read-read dependency
    bnode_later.reset_write_segs()
    bnode_later.reset_read_segs()
    bnode_later.add_read_segs(seg1)
    assert bnode_later.depend_on(bnode_early) is False
    # test write-read dependency
    bnode_early.reset_read_segs()
    bnode_early.reset_write_segs()
    bnode_early.add_write_segs(seg1)
    assert bnode_later.depend_on(bnode_early) is True
    # test write-write dependency
    bnode_later.reset_read_segs()
    bnode_later.reset_write_segs()
    bnode_later.add_write_segs(seg1)
    assert bnode_later.depend_on(bnode_early) is True


def test_node_match_conds():
    bnode = Node()
    seg1 = DataSegment(1)
    bnode.name = "bnode"
    bnode.add_write_segs(seg1)
    assert bnode.match_conds({
        'name': 'bnode'
    }) is True
    assert bnode.match_conds({
        'name': 'bnode', 'w_segs': [seg1]
    }) is True
    assert bnode.match_conds({
        'name': 'wrong', 'w_segs': [seg1]
    }) is False


def test_trans_node_init():
    seg = DataSegment(1)
    # test send
    tnode_send = TransNode(TransNodeType.SEND, sendbuf=seg, dst=2)
    assert tnode_send.op == TransNodeType.SEND
    assert tnode_send.r_segs[0] == seg
    assert len(tnode_send.w_segs) == 0
    # test recv
    tnode_recv = TransNode(TransNodeType.RECV, recvbuf=seg, src=2)
    assert tnode_recv.op == TransNodeType.RECV
    # test fail case
    with pytest.raises(TypeError):
        _ = TransNode(CompNodeType.ADD, sendbuf=seg, src=1)
    with pytest.raises(CommDSLRuntimeError):
        _ = TransNode(TransNodeType.SEND, sendbuf=seg, src=1)
    with pytest.raises(CommDSLRuntimeError):
        _ = TransNode(TransNodeType.RECV, sendbuf=seg, src=2)


def test_trans_node_reduction():
    seg = DataSegment(1)
    tnode_recv = TransNode(TransNodeType.RECV, recvbuf=seg, src=2)
    tnode_recv.reduction = CompNodeType.SUB
    assert tnode_recv.reduction == CompNodeType.SUB

    with pytest.raises(TypeError):
        tnode_recv.reduction = TransNodeType.SEND

    tnode_send = TransNode(TransNodeType.SEND, sendbuf=seg, dst=2)
    assert tnode_send.reduction is None
    with pytest.raises(CommDSLRuntimeError):
        tnode_send.reduction = CompNodeType.ADD


def test_trans_node_repr():
    seg = DataSegment(1)
    tnode = TransNode(TransNodeType.SEND, sendbuf=seg, dst=1)
    assert repr(tnode) == 'Send {} -> rank 1'.format(seg)


def test_comp_node_init():
    seg_lhs = DataSegment(1)
    seg_rhs = DataSegment(1)
    seg_out = DataSegment(1)
    cnode = CompNode(CompNodeType.ADD, seg_lhs, seg_rhs, seg_out, False)
    assert len(cnode.r_segs) == 2
    assert seg_lhs in cnode.r_segs and seg_rhs in cnode.r_segs
    assert len(cnode.w_segs) == 1
    assert cnode.w_segs[0] == seg_out
    assert cnode.lhs == seg_lhs
    assert cnode.rhs == seg_rhs
    assert cnode.output == seg_out


def test_comp_node_inplace():
    seg_lhs = DataSegment(1)
    seg_rhs = DataSegment(1)
    seg_out = DataSegment(1)
    cnode = CompNode(CompNodeType.ADD, seg_out, seg_rhs, seg_out, True)
    assert len(cnode.r_segs) == 1 and cnode.r_segs[0] == seg_rhs
    cnode = CompNode(CompNodeType.ADD, None, seg_rhs, seg_out, True)
    assert len(cnode.r_segs) == 1 and cnode.r_segs[0] == seg_rhs
    cnode = CompNode(CompNodeType.ADD, seg_lhs, seg_out, seg_out, True)
    assert len(cnode.r_segs) == 1 and cnode.r_segs[0] == seg_lhs
    cnode = CompNode(CompNodeType.ADD, seg_lhs, None, seg_out, True)
    assert len(cnode.r_segs) == 1 and cnode.r_segs[0] == seg_lhs
    with pytest.raises(CommDSLRuntimeError):
        _ = CompNode(CompNodeType.ADD,
                     seg_lhs, seg_rhs, seg_out, True)


def test_comp_node_outplace():
    seg_lhs = DataSegment(1)
    seg_rhs = DataSegment(1)
    seg_out = DataSegment(1)
    # outplacement add
    cnode = CompNode(CompNodeType.ADD, seg_lhs, seg_rhs, seg_out, False)
    assert cnode.op == CompNodeType.ADD
    assert cnode.lhs == seg_lhs
    assert cnode.rhs == seg_rhs
    assert cnode.output == seg_out
    assert len(cnode.r_segs) == 2 and cnode.r_segs[0] == seg_lhs
    assert len(cnode.w_segs) == 1 and cnode.w_segs[0] == seg_out

    cnode = CompNode(CompNodeType.ADD, seg_lhs, seg_rhs, seg_lhs, False)
    assert len(cnode.r_segs) == 2 and cnode.r_segs[0] == seg_lhs
    assert cnode.output == seg_lhs


def test_comp_node_repr():
    seg_rhs = DataSegment(1)
    seg_out = DataSegment(1)
    cnode = CompNode(CompNodeType.ADD, seg_out, seg_rhs, seg_out, True)
    assert repr(cnode) == '{out} <- {out} + {rhs}'.format(
        out=seg_out, rhs=seg_rhs
    )
