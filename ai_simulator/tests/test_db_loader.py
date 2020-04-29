import os
from profiler.database_loader import DatabaseLoader
from profiler.database_backend import DatabaseBackendLocalFile

'''
Test cases:
1. normal query, in db
2. wrong op attr, not in db
3. query with disordered attr, in db
4. query with disordered attr, not in db
5. query with ignored attr, in db
6. query with ignored attr, not in db
7. query with not-string input, in db
8. query with not-string input, not in db
9. query with not-string attr_value, in db
10. query with not-string attr_value, not in db
'''


class TestDBLoader():
    def setup_class(self):
        db_path = os.path.join(os.path.dirname(
            __file__), "test_db_loader_input/test_db.json")
        self.database = DatabaseLoader(
            db_type=DatabaseBackendLocalFile, db_file_path=db_path)

    def test_dbloader_normal_yes(self):
        # Test case 1: normal query, in db
        query_op = 'OP_A'
        query_input = []
        query_attr = []
        query_input.append([1, 1, 1, 1])
        query_input.append([256])
        query_attr.append(('attr_0', 'attr_0_value'))
        query_attr.append(('attr_1', 'attr_1_value'))
        result = self.database.search_record(query_op, query_input, query_attr)
        assert result['avg'] == 10.0
        assert result['std_err'] == 0.1

    def test_dbloader_wrong_attr_no(self):
        # Test case 2: wrong op attr, not in db
        query_op = 'OP_B'
        query_input = []
        query_attr = []
        query_input.append([1, 1, 1, 1])
        query_input.append([256])
        query_attr.append(('attr_0', 'attr_0_value'))
        query_attr.append(('attr_1', 'attr_1_value'))
        result = self.database.search_record(query_op, query_input, query_attr)
        assert result is None

    def test_dbloader_disorder_attr_yes(self):
        # Test case 3: query with disordered attr, in db
        query_op = 'OP_C'
        query_input = []
        query_attr = []
        query_attr.append(('attr_1', 'attr_1_value'))
        query_attr.append(('attr_0', '65535'))
        result = self.database.search_record(query_op, query_input, query_attr)
        assert result['avg'] == 30.0
        assert result['std_err'] == 0.3

    def test_dbloader_disorder_attr_no(self):
        # Test case 4. query with disordered attr, not in db
        query_op = 'OP_C'
        query_input = []
        query_attr = []
        query_attr.append(('attr_2', 'attr_2_value'))
        query_attr.append(('attr_0', '65535'))
        result = self.database.search_record(query_op, query_input, query_attr)
        assert result is None

    def test_dbloader_ignore_attr_yes(self):
        # Test case 5. query with ignored attr, in db
        query_op = 'OP_A'
        query_input = []
        query_attr = []
        query_input.append([1, 1, 1, 1])
        query_input.append([256])
        query_attr.append(('attr_0', 'attr_0_value'))
        query_attr.append(('attr_1', 'attr_1_value'))
        query_attr.append(('_class', 'foo'))
        result = self.database.search_record(query_op, query_input, query_attr)
        assert result['avg'] == 10.0
        assert result['std_err'] == 0.1

    def test_dbloader_ignore_attr_no(self):
        # Test case 6. query with ignored attr, not in db
        query_op = 'OP_A'
        query_input = []
        query_attr = []
        query_input.append([1, 1, 1, 2])
        query_input.append([256])
        query_attr.append(('attr_0', 'attr_0_value'))
        query_attr.append(('attr_1', 'attr_1_value'))
        query_attr.append(('_class', 'foo'))
        result = self.database.search_record(query_op, query_input, query_attr)
        assert result is None

    def test_dbloader_no_str_attr_yes(self):
        # Test case 7. query with not-string attr_value, in db
        query_op = 'OP_C'
        query_input = []
        query_attr = []
        query_attr.append(('attr_1', 'attr_1_value'))
        query_attr.append(('attr_0', 65535))
        result = self.database.search_record(query_op, query_input, query_attr)
        assert result['avg'] == 30.0
        assert result['std_err'] == 0.3

    def test_dbloader_no_str_attr_no(self):
        # Test case 8. query with not-string attr_value, not in db
        query_op = 'OP_C'
        query_input = []
        query_attr = []
        query_attr.append(('attr_1', 'attr_1_value'))
        query_attr.append(('attr_0', 65536))
        result = self.database.search_record(query_op, query_input, query_attr)
        assert result is None
