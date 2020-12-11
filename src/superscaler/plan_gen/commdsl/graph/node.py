from superscaler.plan_gen.commdsl.graph.meta import TransNodeType
from superscaler.plan_gen.commdsl.graph.meta import CompNodeType
from superscaler.plan_gen.commdsl.graph.segment import DataSegment
from superscaler.plan_gen.commdsl.errors import CommDSLRuntimeError


class Node:

    """
    Basic class for node in depdendency graph

    Attributes:
        name (str): name of Node
        op (TranNodeType, CommNodeType): op type
        r_segs (list[DataSegment]): list of read buffers
        w_segs (list[DataSegment]): list of write buffers
    """

    def __init__(self):

        self.__name = None
        self.__op = None
        self.__r_segs = []
        self.__w_segs = []

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, val):
        """
        Set node name

        Args:
            val (str): new node name

        Returns:
            None
        """
        if not isinstance(val, str):
            raise TypeError("Node name expected to be a str")
        self.__name = val

    @property
    def op(self):
        """
        Get op type of this node

        Args:
            None

        Returns:
            type (TransNodeType, CompNodeType)
        """
        return self.__op

    @op.setter
    def op(self, op_type):
        """
        Set op type

        Args:
            op_type (TransNodeType, CompNodeType): op type

        Returns:
            None
        """
        if not isinstance(op_type, (TransNodeType, CompNodeType)):
            raise TypeError("Node op type expected to be a node type")
        self.__op = op_type

    @property
    def r_segs(self):
        """
        Get read data segment list of this node

        Args:
            None

        Returns:
            read data segments (list[DataSegment])
        """
        return self.__r_segs

    @r_segs.setter
    def r_segs(self, val):
        """
        Not allow for setting r_segs directly.

        Args:
            val (Any): value to reset r_segs

        Returns:
            None (will raise Error)
        """
        raise CommDSLRuntimeError((
            "Not allowed for setting read data segments directly. Please "
            "refer to add_read_segs and reset_read_segs"))

    @property
    def w_segs(self):
        """
        Get write data segment list of this node

        Args:
            None

        Returns:
            write data segments (list[DataSegment])
        """
        return self.__w_segs

    @w_segs.setter
    def w_segs(self, val):
        """
        Not allow for setting w_segs directly.

        Args:
            val (Any): value to reset w_segs

        Returns:
            None (will raise Error)
        """
        raise CommDSLRuntimeError((
            "Not allowed for setting write data segments directly. Please "
            "refer to add_write_segs and reset_write_segs"))

    def add_read_segs(self, r_segs):
        """
        Append read data segments

        Args:
          r_segs (list, DataSegment): data segment instance or the list of that

        Returns:
          None
        """
        if isinstance(r_segs, list):
            if not all(isinstance(seg, DataSegment) for seg in r_segs):
                raise TypeError(
                    'Type expected to be list[{DataSegment.__name__}]')
            self.__r_segs += r_segs
        else:
            if not isinstance(r_segs, DataSegment):
                raise TypeError(
                    f'Type expected to be {DataSegment.__name__}')
            self.__r_segs.append(r_segs)

    def reset_read_segs(self):
        """
        Reset read data segments to empty

        Args:
            None

        Returns:
            None
        """
        self.__r_segs = []

    def add_write_segs(self, w_segs):
        """
        Append write data segements

        Args:
            w_segs (list[DataSegment], DataSegment):
                DataSegment instance or list of DataSegment

        Returns:
          None
        """
        if isinstance(w_segs, list):
            if not all(isinstance(seg, DataSegment) for seg in w_segs):
                raise TypeError(
                    f'Type expected to be list[{DataSegment.__name__}]')
            self.__w_segs += w_segs
        else:
            if not isinstance(w_segs, DataSegment):
                raise TypeError(
                    f"Type expected to be {DataSegment.__name__}")
            self.__w_segs.append(w_segs)

    def reset_write_segs(self):
        """
        Reset write data segments to empty

        Args:
            None

        Returns:
            None
        """
        self.__w_segs = []

    def depend_on(self, other):
        """
        Dependency check for this node with other node. `other` node will
        be earlier than this node in partial order

        Args:
            other (Node)

        Returns:
            return True if `other` should be executed before self node,
            else False
        """
        if not isinstance(other, Node):
            raise TypeError("Type expected to be a Node")
        for rseg in other.r_segs:
            # read-write dependency
            for wseg in self.w_segs:
                if rseg.overlap(wseg):
                    return True
        for wseg in other.w_segs:
            # write-read / write dependency
            for myseg in self.r_segs + self.w_segs:
                if wseg.overlap(myseg):
                    return True
        return False

    def match_conds(self, conds):
        """
        Return True if condition (conds) matches.
        The conds is specified as a dict of attributes name and value.

        Args:
            conds (dist{attr: val}):
                condition specified as attributes (str)
                and expected value (Any).

        Returns:
            True if this node matches all conditions
        """
        if not isinstance(conds, dict):
            raise TypeError("`conds` type expected to be a dict")
        for attr in conds:
            if hasattr(self, attr):
                if getattr(self, attr) != conds[attr]:
                    return False
            else:
                return False
        return True

    def __repr__(self):
        """
        Node information string

        Args:
            None

        Returns:
            string (str)
        """
        return 'BaseNode'


class TransNode(Node):
    """
    Represent the communication primitive node.

    Attributes:
        name (str): name of Node
        op (TranNodeType, CommNodeType): op type
        r_segs (list[DataSegment]): list of read buffers
        w_segs (list[DataSegment]): list of write buffers
        reduction (None, CommNodeType): reduction op type on recv
    """

    def __init__(self, op_type,
                 sendbuf=None, dst=None,
                 recvbuf=None, src=None, reduction=CompNodeType.COPY):
        """
        Create trans node (communication primitive node)

        Args:
            op_type (TransNodeType): send / recv
            sendbuf (DataSegment): send buffer
            dst (int): destination node
            recvbuf (DataSegment): receive buffer
            reduction (CompNodeType): reduction op when recving data
        """
        super().__init__()
        if not isinstance(op_type, TransNodeType):
            raise TypeError(
                f'Expected op_type to be {TransNodeType.__name__}')

        self.op = op_type
        if self.op == TransNodeType.SEND:
            if sendbuf is None or dst is None:
                raise CommDSLRuntimeError(
                    "send op must provide sendbuf and dst")
            self.add_read_segs(sendbuf)
            self.dst = dst
            self.__reduction = None
        elif self.op == TransNodeType.RECV:
            if recvbuf is None or src is None:
                raise CommDSLRuntimeError(
                    "recv op must provide recvbuf and src")
            self.add_write_segs(recvbuf)
            self.src = src
            self.reduction = reduction
        else:
            raise TypeError(f'Unknown {TransNode.__name__} op type')

    @property
    def dst(self):
        """
        Get destination value

        Args:
            None

        Returns:
            rank (int)
        """
        if self.op == TransNodeType.SEND:
            return self.__dst
        else:
            raise CommDSLRuntimeError(
                "Try to access dst val from a non-send node")

    @dst.setter
    def dst(self, rank):
        """
        Set destination value

        Args:
            rank (int): destination rank for the send node

        Returns:
            None
        """
        if self.op != TransNodeType.SEND:
            raise CommDSLRuntimeError(
                "Try to set dst val on a non-send node")
        if not isinstance(rank, int):
            raise TypeError("dst expected to be an int")
        self.__dst = rank

    @property
    def src(self):
        """
        Get destination value

        Args:
            None

        Returns:
            rank (int)
        """
        if self.op == TransNodeType.RECV:
            return self.__dst
        else:
            raise CommDSLRuntimeError(
                "Try to access src val from a non-recv node")

    @src.setter
    def src(self, rank):
        """
        Set destination value

        Args:
            rank (int): source rank for the recv node

        Returns:
            None
        """
        if self.op != TransNodeType.RECV:
            raise CommDSLRuntimeError(
                "Try to set src val on a non-recv node")
        if not isinstance(rank, int):
            raise TypeError("src expected to be an int")
        self.__src = rank

    @property
    def reduction(self):
        """
        Get reduction value.

        Note send node will always return a None.

        Args:
            None

        Returns:
            reduction op type (None, CompNodeType)
        """
        return self.__reduction

    @reduction.setter
    def reduction(self, reduction_op):
        """
        Reset reduction op.

        Args:
            reduction_op (TransNodeType, None): new reduction op

        Returns:
            None
        """
        if not isinstance(reduction_op, CompNodeType):
            raise TypeError("Must be a computation node type")
        if self.op != TransNodeType.RECV:
            raise CommDSLRuntimeError(
                "Trying to set reduction on non-recv op")
        self.__reduction = reduction_op

    def __repr__(self):
        if self.op == TransNodeType.SEND:
            rseg = self.r_segs[0]
            return 'Send {} -> rank {}'.format(rseg, self.dst)
        elif self.op == TransNodeType.RECV:
            wseg = self.w_segs[0]
            return 'Recv {} <- rank {} Reduction: {}'\
                .format(wseg, self.src, self.reduction)
        else:
            raise TypeError(
                "Unknown type for TransNode: {}".format(self.op))


class CompNode(Node):
    """
    Represent the computation (on data segment) node.

    Attributes:
        name (str): name of Node
        op (TranNodeType, CommNodeType): op type
        r_segs (list[DataSegment]): list of read buffers
        w_segs (list[DataSegment]): list of write buffers
        lhs (DataSegment, int, float): left hand side value
        rhs (DataSegment, int, float): right hand side value
        output (DataSegment): output data segment
    """

    def __init__(self, op_type, lhs, rhs, output, inplace):
        """
        Create data segment operation node.
        To create a node with inplacement,
        at least one of lhs or rhs is None,
        or one of lhs or rhs is equal to output

        Args:
            op (CompNodeType): operation of +,-,*,/
            lhs (DataSegment, int, float or None): left hand side value.
            rhs (DataSegment, int, float or None): right hand side value.
                Note should be at least one DataSegment for lhs and rhs.
            output (DataSegment): output DataSegment
        """
        super().__init__()
        if not isinstance(op_type, CompNodeType):
            raise TypeError(
                f'Expected op_type to be {CompNodeType.__name__}')

        self.op = op_type
        self.lhs = lhs
        self.rhs = rhs
        self.output = output
        if inplace:
            self._set_inplacement()
        else:
            self._set_outplacement()

    @property
    def lhs(self):
        """
        Get lhs of this node

        Args:
            None

        Returns:
            lhs (DataSegment, int, float, None)
        """
        return self.__lhs

    @lhs.setter
    def lhs(self, val):
        """
        Set lhs of this node

        Args:
            val (DataSegment, int, float, None): value to set

        Returns:
            None
        """
        if (val is not None) and \
           (not isinstance(val, (DataSegment, int, float))):
            raise TypeError(
                "lhs expected to be one of data segment, int or float")
        self.__lhs = val

    @property
    def rhs(self):
        """
        Get rhs of this node

        Args:
            None

        Returns:
            rhs (DataSegment, int, float, None)
        """
        return self.__rhs

    @rhs.setter
    def rhs(self, val):
        """
        Set rhs of this node

        Args:
            val (DataSegment, int, float, None): value to set

        Returns:
            None
        """
        if (val is not None) and \
           (not isinstance(val, (DataSegment, int, float))):
            raise TypeError(
                "rhs expected to be one of data segment, int or float")
        self.__rhs = val

    @property
    def output(self):
        """
        Get output of this node

        Args:
            None

        Returns:
            output (DataSegment)
        """
        return self.__output

    @output.setter
    def output(self, output):
        """
        Set output of this node

        Args:
            output (DataSegment, int, float, None)

        Returns:
            None
        """
        if not isinstance(output, DataSegment):
            raise TypeError("output expected to be a data segment")
        self.__output = output

    def _set_inplacement(self):
        """
        Assign data segment to inplacement update pattern.

        Require at least one of lhs or rhs is None,
        or is equal to output.

        Args:
            None

        Returns:
            None
        """
        self.add_write_segs(self.output)
        if self.lhs is None or \
                (isinstance(self.lhs, DataSegment) and
                 self.lhs == self.output):
            if isinstance(self.rhs, DataSegment):
                self.add_read_segs(self.rhs)
        elif self.rhs is None or \
                (isinstance(self.rhs, DataSegment) and
                 self.rhs == self.output):
            if isinstance(self.lhs, DataSegment):
                self.add_read_segs(self.lhs)
        else:
            raise CommDSLRuntimeError(
                "Not an inplacement-update compute op.")

    def _set_outplacement(self):
        """
        Assign data segment to outplacement update pattern.

        Args:
            None

        Returns:
            None
        """
        self.add_write_segs(self.output)
        if isinstance(self.lhs, DataSegment):
            self.add_read_segs(self.lhs)
        if isinstance(self.rhs, DataSegment):
            self.add_read_segs(self.rhs)

    def __repr__(self):
        """
        CompNode information string

        Args:
          None

        Returns:
          string
        """
        op_type = None
        if self.op == CompNodeType.ADD:
            op_type = '+'
        elif self.op == CompNodeType.SUB:
            op_type = '-'
        elif self.op == CompNodeType.MUL:
            op_type = '*'
        elif self.op == CompNodeType.DIV:
            op_type = '/'
        elif self.op == CompNodeType.COPY:
            op_type = ''
        elif self.op == CompNodeType.CREATE:
            op_type = 'create'
        else:
            raise TypeError(
                "Unknown type for compute operation {}".format(self.op))

        lhs = '' if self.lhs is None else self.lhs
        return '{} <- {} {} {}'.format(
            self.output, lhs, op_type, self.rhs)
