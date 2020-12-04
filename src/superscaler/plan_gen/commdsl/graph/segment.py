# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.commdsl.graph.segment_pool import DataSegmentPool
from superscaler.plan_gen.commdsl.errors import CommDSLRuntimeError


class DataSegment:
    """
    DataSegment class for defining a data segment.

    DataSegment consists of multiple chunks. Each chunk has multiple blocks.

    Block is the basic data unit of a data segment.
    Every block has the same capability to contain the same number of
    real data elements.

    The goal of DataSegment base class is for auto bnum scaling, to make
    the bnum always an integer when calling `DataSegment.slice(K)`
    and meanwhile keep inter-data segment bnum comparable.

    Suppose a data segment is sliced to K chunks,
    bnum should guarantee two properties:
      -- intra-data segments (data segment and its sliced data segments):
            bnum % K == 0 => minimal dividable, each chunk has
            an integer number of blocks
      -- inter-data segments (data segments with different ids):
            block number should be consistent => simultaneously scale
            all block numbers, and meanwhile keep
            intra-data segment progerty for each data segment

    Properties for sliced data segment chunks:
        Each chunk has the same name with its parent (root) segment;
        Each chunk has the same id with its parent (root) segment;

    Attributes:
        bnum (int): block number
        cnum (int): chunk number (set by DataSegment.slice)
        name (str): data segment name
        root_bnum (int): root data segment bnum
        root_blk_begin (int): start offset of this DataSegment on root space
        root_blk_end (int): end offset of this DataSegment on root space.
            A data segment range in root is [root_start, root_end)
        _id (int): data segment id
    """

    def __init__(self, bnum, parent=None, start_cid=None, name=None):
        """
        Initialize the data segment with specified bnum

        Args:
            bnum (int): virtual bnum
                should be comparable among all data segment instances
            parent (DataSegment): parent DataSegment
            start_cid (int): chunk idx for parent data segment
            name (str): name of the data segment
        """
        if (not isinstance(parent, DataSegment)) and (parent is not None):
            raise TypeError(
                "parent should be None or DataSegment instance")
        if not isinstance(bnum, int):
            raise TypeError("bnum should be a positive integer")
        if bnum <= 0:
            raise CommDSLRuntimeError("bnum should be a positive integer")
        self.bnum = bnum
        self.cnum = 1
        if parent is None:
            self.root_bnum = bnum
            self.root_blk_begin = 0
            self.root_blk_end = bnum
            self._id = DataSegmentPool().gen_id()
            self.name = name
            if name is None:
                self.name = 'data_' + str(self._id)
        else:
            chunk_blocks = parent.bnum // parent.cnum
            self.root_bnum = parent.root_bnum
            self.root_blk_begin = \
                parent.root_blk_begin + chunk_blocks * start_cid
            self.root_blk_end = self.root_blk_begin + bnum
            self._id = parent._id
            self.name = parent.name
        # register to pool
        DataSegmentPool().register(self)

    @property
    def bnum(self):
        """
        Get block number

        Returns:
            bnum (int)
        """
        return self._bnum

    @bnum.setter
    def bnum(self, new_block_num):
        """
        Set block number

        Args:
          bnum (int): number of chunks

        Return:
          None
        """
        if '_bnum' not in self.__dict__:
            self._bnum = new_block_num
        else:
            if new_block_num < self._bnum:
                raise CommDSLRuntimeError((
                    "DataSegment block num "
                    "should progressively increase."))
            if new_block_num % self._bnum != 0:
                raise CommDSLRuntimeError((
                    "DataSegment new block num should be "
                    "dividable by old block num."))
            factor = new_block_num // self._bnum
            self._bnum = new_block_num
            self.root_bnum *= factor
            self.root_blk_begin *= factor
            self.root_blk_end *= factor

    def get_id(self):
        """
        Get data segment id.

        Args:
            None

        Return:
            DataSegment id (int)
        """
        return self._id

    def slice(self, num):
        """
        Slice the data segment to `num` of chunks

        Args:
          num (int): number of chunks

        Return:
          None
        """
        if self.bnum % num != 0:
            DataSegmentPool().scale_size(num)
        self.cnum = num

    def scale(self, factor):
        """
        Scale block number by factor.
        Note all the number and offsets will be scaled
        to keep consistent semantics except chunk number.

        Args:
            factor (int): scale factor

        Returns:
            None
        """
        if not isinstance(factor, int):
            raise TypeError("scale can only accept Integer factor")
        self.bnum = self.bnum * factor

    def overlap(self, other):
        """
        Check whether overlapped with other data segment

        Args:
            other (DataSegment)

        Returns:
            True if two data segments have intersection,
            else False
        """
        if other._id == self._id:
            if max(self.root_blk_begin, other.root_blk_begin) < \
               min(self.root_blk_end, other.root_blk_end):
                return True
        return False

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += self.cnum
            if key < 0 or key >= self.cnum:
                raise IndexError("Index out of sliced chunk number")
            return DataSegment(
                self.bnum // self.cnum,
                parent=self,
                start_cid=key
            )
        elif isinstance(key, slice):
            if key.step is not None:
                raise IndexError("Not support of stepped index")
            start = key.start if key.start >= 0 else key.start + self.cnum
            stop = key.stop if key.stop >= 0 else key.stop + self.cnum
            if start < 0 or start > self.cnum or \
               stop < 0 or stop > self.cnum:
                raise IndexError("Index out of sliced chunk number")
            chunk_num = stop - start
            if chunk_num < 0:
                raise IndexError("Cannot slice chunks in reverse order")
            return DataSegment(
                self.bnum // self.cnum * chunk_num,
                parent=self,
                start_cid=start
            )
        else:
            raise TypeError("Not supported indexing type")

    def __setitem__(self, *args):
        return

    def __eq__(self, other):
        """
        Compare is two data segment instance is equal:
        all attr should be same except chunk bnum

        Args:
          other (DataSegment)
        """
        if not isinstance(other, self.__class__):
            raise TypeError("Only support compare with DataSegment")
        for attr_name in self.__dict__:
            if attr_name == 'cnum':
                continue
            attr_val = self.__dict__[attr_name]
            if not hasattr(other, attr_name):
                return False
            elif other.__dict__[attr_name] != attr_val:
                return False
        return True

    def __repr__(self):
        """
        Print function for getting internal information from. Use print(seg).

        Args:
          None

        Returns:
          None
        """
        return 'Data-{}:({})[{}:{}]'.format(
            self._id, self.bnum, self.root_blk_begin, self.root_blk_end
        )

    # data segment operation interface
    def __add__(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")

    def __iadd__(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")

    def __sub__(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")

    def __isub__(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")

    def __mul__(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")

    def __imul__(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")

    def __truediv__(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")

    def __itruediv__(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")

    def copy(self, other):
        raise CommDSLRuntimeError("Error call to base DataSegment class.")
