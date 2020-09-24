import pytest
import json
from plan.parser.profiler.database_loader import DatabaseLoader
from plan.parser.profiler.database_backend import DatabaseBackendException

TEST_NODELIST_FILE = 'tests/data/tf_parser_testbench/db_nodelist.json'
TEST_DB_FILE = 'tests/data/tf_parser_testbench/db_test.json'


def test_database_loader():
    # Test incorrect input path
    with pytest.raises(DatabaseBackendException):
        database_loader = DatabaseLoader(db_file_path="incorrect")

    # Test None input path
    with pytest.raises(DatabaseBackendException):
        database_loader = DatabaseLoader(db_file_path=None)

    # Test wrong db type input path
    with pytest.raises(DatabaseBackendException):
        database_loader = DatabaseLoader(db_type=None,
                                         db_file_path=TEST_DB_FILE)

    # 'Start test db loader'
    database_loader = DatabaseLoader(db_file_path=TEST_DB_FILE)
    nodelist = json.load(open(TEST_NODELIST_FILE, 'r'))

    # Test search_record function
    node = nodelist[0]
    result = database_loader.search_record(node['op'],
                                           node['input_shapes'],
                                           node['attr_list'])
    assert(result['avg'] == 10.0)

    # Test search_record function with incorrect input
    result = database_loader.search_record(None,
                                           None,
                                           None)
    assert(result is None)

    # Test add_record function for revising
    node = nodelist[0]
    result_dict = {
        "avg": 20.0,
        "std_err": 0.1
    }
    database_loader.add_record(node['op'],
                               node['input_shapes'],
                               node['attr_list'],
                               result_dict)
    result = database_loader.search_record(node['op'],
                                           node['input_shapes'],
                                           node['attr_list'])
    assert(result['avg'] == 20.0)

    # Test remove_record function for removing old record
    node = nodelist[0]
    database_loader.remove_record(node['op'],
                                  node['input_shapes'],
                                  node['attr_list'])
    result = database_loader.search_record(node['op'],
                                           node['input_shapes'],
                                           node['attr_list'])
    assert(result is None)

    # Test add_record function for adding new record
    node = nodelist[0]
    result_dict = {
        "avg": 10.0,
        "std_err": 0.1
    }
    database_loader.add_record(node['op'],
                               node['input_shapes'],
                               node['attr_list'],
                               result_dict)
    result = database_loader.search_record(node['op'],
                                           node['input_shapes'],
                                           node['attr_list'])
    assert(result['avg'] == 10.0)
