from profiler.profiler import Profiler

def test_profiler():
    op_profiler = Profiler()
    assert op_profiler.get_node_execution_time(object) == 10