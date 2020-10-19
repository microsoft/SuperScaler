import os
import errno
# from scaler_graph import ...
# from plan_gen import ResourcePool, TFParser, PlanGenerator
# from runtime import ...


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
        self._cache_dir = os.path.join(os.path.expanduser('~'), 'tmp/')
        self._is_initialized = False

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

    def __create_cache_dir(self):
        """ A function that create cache directory
            if the cache directory exist, did nothing.
        """
        # Check whether the cache_dir exits and is legal.
        if not isinstance(self._cache_dir, str):
            raise OSError
        elif not os.path.exists(self._cache_dir):
            try:
                os.mkdir(self._cache_dir)
            except OSError as exc:
                if exc.errno == errno.EEXIST and\
                   os.path.isdir(self._cache_dir):
                    pass
                else:
                    raise

    def is_initialized(self):
        """ Returns True if Superscaler is initialized """
        return self._is_initialized

    def init(self, user_model, deployment_plan, strategy, resource_pool):
        """ A function that initializes Superscaler.

        Args:
          user_model: Model description provided by user. Currently
            Superscaler supports TensorFlow model and NNfusion model.
          deployment_plan: List specifying devices for distributed deployment.
          strategy: distributed training strategy including data parallelism,
            model parallelism and pipeline parallelism.
          resource_pool: JSON file specifying hardware description and network
            topology.
        """
        try:
            self.__create_cache_dir()
            self.__init_partition_graphs(user_model, deployment_plan, strategy)
            self.__init_communication_plan(resource_pool)
            self.__init_runtime_setting()
            self._is_initialized = True
        except SuperscalerError:
            raise SuperscalerError("Superscaler initialization failed")

    def __init_partition_graphs(self, user_model, deployment_plan, strategy):
        # TODO scaler_graph support
        pass

    def __init_communication_plan(self, resource_pool):
        # TODO plan_gen support
        pass

    def __init_runtime_setting(self):
        # TODO runtime support
        pass

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
