class Edge:
    '''
    The edge
    '''
    _ID = 0

    def __init__(self, src_node, src_idx, dest_node, dest_idx):
        self.id = self._ID
        self.__class__._ID += 1
        self.src_node = src_node
        self.src_idx = src_idx
        self.dest_node = dest_node
        self.dest_idx = dest_idx

    def is_control_edge(self):
        return self.src_idx == -1
