import os
from adapter.tf_adapter import TFAdapter


def test_tf_adapter():
    test_tf_adapter = TFAdapter()
    tf_adapter_unit_test_file_relative_path\
        = os.path.join(
            os.path.dirname(__file__),
            "test_tf_adapter_input", "tf_adapter_unit_test.pbtxt")
    tf_adapter_unit_test_file_absolute_path\
        = os.path.abspath(
            os.path.expanduser(tf_adapter_unit_test_file_relative_path))
    test_tf_graph_def = test_tf_adapter.load_pbtxt_file(
        tf_adapter_unit_test_file_absolute_path)
    test_parsed_tf_graph = test_tf_adapter.parse_protobuf_graph(
        test_tf_graph_def)
    assert len(test_parsed_tf_graph) == 9
