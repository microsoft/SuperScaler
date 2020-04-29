from .database_loader import DatabaseLoader


class Profiler(object):
    def __init__(self):
        pass

    def get_node_execution_time(self, node):
        return 10


class TFProfiler(Profiler):
    def __init__(self, db_type='default', **kwargs):
        self.database = DatabaseLoader(db_type, **kwargs)

    def get_node_execution_time(self, node):
        # TODO
        op = node.op
        input_shape_list = node.input_shape_list
        attr_list = node.attr_list
        result = self.database.search_record(op, input_shape_list, attr_list)
        return result
