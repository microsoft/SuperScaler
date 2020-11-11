import json
import tensorflow as tf
from google.protobuf import text_format
from frontend.runtime.runtime import Runtime


class TFRuntime(Runtime):
    def __init__(self, graph_file, graph_desc_file, plan_path, lib_path):
        """A function that initializes TFRuntime.
        Args:
          graph_file: string specifying the path where tensroflow graph is.
          graph_desc_file: string specifying the path where graph_desc is.
          plan_path: string specifying the path where communication plan is.
          lib_path: string specifying the path where runtime library is.
        """
        self.graph_file = graph_file
        self.graph_desc_file = graph_desc_file
        super().__init__(plan_path, lib_path)

        try:
            self.tf_exe_lib = tf.load_op_library(self.get_sc_lib_path())
        except BaseException:
            raise Exception("libsuperscaler: %s is not loaded by tensorflow!" %
                            (self.libsuperscaler))
        self._graph, self._inits, self._feeds, self._fetches, self._targets =\
            self.__load_tf_graph(graph_file, graph_desc_file, self.device_id())

    def __load_tf_graph(self, graph_file, graph_desc_file, device_id):
        """A function that loads tensorflow graph and extract runtime infos
        Args:
          graph_file: string specifying the path where tensroflow graph is.
          graph_desc_file: string specifying the path where graph_desc is.
          device_id: string specifying the running devices
        """

        # open tf_graph
        with open(graph_file) as f:
            graph_txt = f.read()
        graph_def = text_format.Parse(graph_txt, tf.compat.v1.GraphDef())
        graph_clone = tf.Graph()

        # extract key tensors and operatos from tf_graph by graph_desc
        with graph_clone.as_default():
            with tf.device('/gpu:{}'.format(device_id)):
                tf.import_graph_def(graph_def=graph_def, name="")
            with open(graph_desc_file, 'r') as f:
                desp = json.load(f)
                inits = [
                    graph_clone.get_operation_by_name(i) for i in desp['inits']
                ]
                # TODO: u should make a dict out of feeds
                feeds = [i for i in desp['feeds']]
                fetches = [
                    graph_clone.get_tensor_by_name(i) for i in desp['fetches']
                ]
                targets = [
                    graph_clone.get_operation_by_name(i)
                    for i in desp['targets']
                ]
                return graph_clone, inits, feeds, fetches, targets

    @property
    def inits(self):
        return self._inits

    @property
    def graph(self):
        return self._graph

    @property
    def feeds(self):
        return self._feeds

    @property
    def fetches(self):
        return self._fetches

    @property
    def targets(self):
        return self._fetches
