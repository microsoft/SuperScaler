import pytest
import json
from plan.adapter.profiler.profiler import TFProfiler
from plan.adapter.profiler.database_backend import DatabaseBackendException

TEST_NODELIST_FILE = 'tests/data/tf_parser_testbench/db_nodelist.json'
TEST_DB_FILE = 'tests/data/tf_parser_testbench/db_test.json'


def test_tf_db():
    # Test incorrect input path
    with pytest.raises(DatabaseBackendException):
        profiler = TFProfiler(db_file_path="incorrect")

    # Test None input path
    with pytest.raises(DatabaseBackendException):
        profiler = TFProfiler(db_file_path=None)

    # Test wrong db type input path
    with pytest.raises(DatabaseBackendException):
        profiler = TFProfiler(db_type=None, db_file_path=TEST_DB_FILE)

    # 'Start test db'
    profiler = TFProfiler(db_file_path=TEST_DB_FILE)
    nodelist = json.load(open(TEST_NODELIST_FILE, 'r'))

    print('Test node 0')
    node = nodelist[0]
    result = profiler.get_node_execution_time(node)
    assert(result['avg'] == 10.0)

    print('Test node 1')
    node = nodelist[1]
    result = profiler.get_node_execution_time(node)
    assert(result['avg'] == 20.0)

    node = None
    result = profiler.get_node_execution_time(node)
    assert(result is None)
    print('Test pass!')
