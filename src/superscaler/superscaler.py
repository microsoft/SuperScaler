# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import errno
import tempfile
from pathlib import Path
from superscaler.plan_gen import ResourcePool, SuperScalerAdapter
from superscaler.runtime import PlanAssigner
import logging


class SuperscalerError(Exception):
    """Exception raised for Superscaler class.

    Attributes:
        expression -- input expression in which the error occurred
    """

    def __init__(self, expression):
        self.expression = expression


class Superscaler(object):
    """ Wrapper class for the basic Superscaler API. """

    def __init__(self):
        """ Superscaler automatically creates a cache directory
            in ~/tmp for storing tmp files.
        """
        super().__init__()
        self._logger = logging.getLogger('Superscaler.event.Event')
        self._logger.setLevel(logging.INFO)
        self._cache_dir = os.path.join(os.path.expanduser('~'), 'tmp/')
        self._working_dir = None
        self._is_initialized = False

        self._resoure_pool = ResourcePool()
        self._plan_adapter = SuperScalerAdapter()
        self._plan_assigner = PlanAssigner()

    def get_working_dir(self):
        return self._working_dir

    def get_cache_dir(self):
        return self._cache_dir

    def set_cache_dir(self, cache_dir):
        """ A function that specifies cache directory
            if the cache directory is not legal, return False.
            if the cache directory is legal, return True.
        Args:
          cache_dir: string specifying the cache directory.
        """

        if not isinstance(cache_dir, str):
            return False
        else:
            self._cache_dir = cache_dir
            return True

    def _create_cache_dir(self, cache_dir):
        """ A function that create cache directory
            if the cache directory exist, did nothing.
        """
        # Check whether the cache_dir exits and is legal.
        if not isinstance(cache_dir, str):
            raise OSError
        elif not os.path.exists(cache_dir):
            try:
                os.mkdir(cache_dir)
            except OSError as exc:
                if exc.errno == errno.EEXIST and\
                   os.path.isdir(cache_dir):
                    pass
                else:
                    raise

    def is_initialized(self):
        """ Returns True if Superscaler is initialized """
        return self._is_initialized

    def init(self, apply_gradient_op, loss, deployment_setting,
             strategy, communication_DSL, resource_pool):
        """ A function that initializes Superscaler.

        Args:
          apply_gradient_op: apply_gradient_op operator from the given graph
          loss: loss tensor from the given graph
          deployment_setting: List specifying for distributed deployment.
          strategy: distributed training strategy including data parallelism,
            model parallelism and pipeline parallelism.
          communication_DSL: domain specific language to decribe communication
          resource_pool: JSON file specifying hardware description and network
            topology.
        """
        # apply_gradient_op, loss and strategy are platform-specified,
        # Checking is done on self._init_partition_graphs function
        if not isinstance(resource_pool, str):
            raise SuperscalerError("resource_pool should be inited from file")
        if not isinstance(communication_DSL, str):
            raise SuperscalerError("communication_DSL should be str")
        if not isinstance(deployment_setting, dict):
            raise SuperscalerError("deployment_setting must be dict")

        try:
            self._create_cache_dir(self._cache_dir)
            self._tempfile = tempfile.TemporaryDirectory(
                dir=self._cache_dir)
            self._working_dir = self._tempfile.name
            self._logger.info("Creates a cache directory at %s \
                for storing tmp files." % (self._working_dir))
            self._init_partition_graphs(apply_gradient_op, loss, strategy)
            self._init_communication_plan(resource_pool, communication_DSL)
            self._init_runtime_setting(deployment_setting)
            self._is_initialized = True
        except SuperscalerError:
            raise SuperscalerError("Superscaler initialization failed")

    def _init_partition_graphs(self, apply_gradient_op, loss, strategy):
        """ A function that partitions graph by parallelism strategy.

        Args:
          apply_gradient_op: apply_gradient_op operator from the given graph
          loss: loss tensor from the given graph
          strategy: distributed training strategy including data parallelism,
            model parallelism and pipeline parallelism.
        """
        self._partition_graphs = []
        self._graph_count = 0
        self._graph_config = {}

    def _init_communication_plan(self, resource_pool, communication_DSL):
        """ A function that generates communication_plan from resource_pool.

        Args:
          resource_pool: JSON file specifying hardware description and network
            topology.
          communication_DSL: domain specific language to decribe communication
        """
        if not isinstance(resource_pool, str):
            raise SuperscalerError("resource_pool should be inited from file")
        if not isinstance(communication_DSL, str):
            raise SuperscalerError("communication_DSL should be str")

        self._communication_plan = []

    def _init_runtime_setting(self, deployment_setting):
        """ A function that create runtime_setting files.

        Args:
          deployment_setting: dict specifying the mapping of hostname and ip.
        """
        if not isinstance(deployment_setting, dict):
            raise SuperscalerError("deployment_setting must be dict")

        self._assigned_plan = self._plan_assigner.assign(
            self._communication_plan, deployment_setting)

        # Dump runtime file into self._working_dir
        for i in range(self._graph_count):
            tmp_rank_dir = os.path.join(self._working_dir, str(i))
            os.mkdir(tmp_rank_dir)

            plan_path = os.path.join(tmp_rank_dir, 'plan.json')
            json.dump(self._assigned_plan[i],
                      open(plan_path, 'w'),
                      indent=4,
                      sort_keys=True)

            graph_config_path = os.path.join(tmp_rank_dir, 'model_desc.json')
            json.dump(self._graph_config,
                      open(graph_config_path, 'w'),
                      indent=4,
                      sort_keys=True)

            partition_graphs_path = os.path.join(tmp_rank_dir, 'graph.pbtxt')
            file = Path(partition_graphs_path)
            file.write_text(self._partition_graphs[i])

    def run(self):
        """ A function that performs distributed training.
            This function is avaliable when self.is_initialized() is True
        """

        if self.is_initialized() is True:
            # TODO runtime support
            """
            runtime.run(graph_path, plan_path)
            """
            pass
        else:
            raise SuperscalerError("Superscaler must be run \
                                    after initialization is complete")
