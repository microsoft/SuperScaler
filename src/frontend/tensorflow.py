from frontend.superscaler import Superscaler, SuperscalerError
from frontend.scaler_graph import tf_adapter, Parallelizer, DataParallelism
from frontend.plan_gen import TFParser, PlanGenerator, SuperScalerAdapter
from frontend.runtime.util import distribute_resources, launch
import tensorflow as tf


class tensorflow(Superscaler):
    """ Wrapper class for the Superscaler API for tensorflow framework. """

    def __init__(self):
        super().__init__()
        self._plan_parser = TFParser()

    def _init_partition_graphs(self, apply_gradient_op, loss, strategy):
        """ A function that partition tensorflow graph by parallelism strategy.

        Args:
          apply_gradient_op: apply_gradient_op operator of tensorflow graph
          loss: loss tensor of tensorflow graph
          strategy: distributed training strategy including data parallelism,
            model parallelism and pipeline parallelism.
        """
        if not isinstance(apply_gradient_op, tf.Operation):
            raise SuperscalerError("apply_gradient_op must be tf.Operation")
        if not isinstance(loss, tf.Tensor):
            raise SuperscalerError("loss must be tf.Tensor")
        if not isinstance(strategy, DataParallelism):
            raise SuperscalerError("Unsupport parallelism strategy")

        # run_parallelisms
        merged_sc_graph = tf_adapter.import_tensorflow_model(
            apply_gradient_op, loss, self._working_dir)
        parallelizer = Parallelizer(merged_sc_graph)
        parallelizer.register_parallelism(strategy)
        parallelizer.run_parallelisms()

        # Convert partition_graphs into tf_protobuf
        self._partition_graphs = []
        for graph in parallelizer.graphs:
            self._partition_graphs.append(
                tf_adapter.export_graph_to_tf_file(graph))
        self._graph_count = len(parallelizer.graphs)
        self._graph_config = tf_adapter.get_tf_runtime_config(merged_sc_graph)

    def _init_communication_plan(self, resource_pool, communication_DSL):
        """ A function that generate communication_plan from resource_pool.

        Args:
          resource_pool: JSON file specifying hardware description and network
            topology.
          communication_DSL: domain specific language to decribe communication
        """
        if not isinstance(resource_pool, str):
            raise SuperscalerError("resource_pool should be inited from file")
        if not isinstance(communication_DSL, str):
            raise SuperscalerError("communication_DSL should be str")

        # init plan_generator
        self._resoure_pool.init_from_yaml(resource_pool)
        devices = ["device_" + str(i) for i in range(self._graph_count)]
        nodelist = self._plan_parser.parse_graphs(
            self._partition_graphs, devices, load_from_memory=True)

        # Generate communication_plan
        plan_generator = PlanGenerator(nodelist, self._resoure_pool)
        plan = plan_generator.get_execution_plan('Allreduce',
                                                 communication_DSL)

        # Adapt plan for Superscaler
        self._plan_adapter = SuperScalerAdapter()
        self._plan_adapter.set_plan(plan)
        self._communication_plan = self._plan_adapter.adapt_plan()

    def run(self):
        """ A function that performs distributed training.
            This function is avaliable when self.is_initialized() is True
        """

        if self.is_initialized() is True:
            deployment_config, rank2ip =\
                self._plan_assigner.get_deployment_config(self._assigned_plan)
            remote_resource_dir = distribute_resources(deployment_config,
                                                       self._working_dir)
            cmd_per_worker = [
                'python -m frontend.runtime.tensorflow.runner '
                '{resource_dir}/{grank}'
                .format(resource_dir=remote_resource_dir, grank=grank)
                for grank, _ in enumerate(rank2ip)
            ]
            launch(rank2ip, cmd_per_worker)
        else:
            raise SuperscalerError("Superscaler must be run \
                                    after initialization is complete")
