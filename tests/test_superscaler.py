# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pytest
import shutil
from superscaler.superscaler import Superscaler, SuperscalerError


def test_superscaler():

    # Init SuperscalerBasics class
    sc = Superscaler()

    # Test for set_cache_dir function
    # None path test
    assert(sc.set_cache_dir(None) is False)
    # Int path test
    assert(sc.set_cache_dir(1) is False)
    cache_dir = os.path.join(os.path.dirname(__file__), 'tmp/')
    assert(sc.set_cache_dir(cache_dir) is True)

    # Test for get_working_dir function
    assert(sc.get_working_dir() is None)

    # Test for error by calling sc.run() before initialization is complete
    with pytest.raises(SuperscalerError):
        sc.run()

    # Test for is_initialized() function, default setting is false
    assert(sc.is_initialized() is False)

    # We didn't do any real job on the virtual Superscaler,
    # Therefore we just use fake input for testing.
    apply_gradient_op = "No check for apply_gradient_op on Superscaler"
    loss = "No check for loss on Superscaler"
    strategy = "No check for strategy on Superscaler"

    # We can test the deployment_setting and resource_pool here.
    deployment_setting = {"1": "10.0.0.21"}
    resource_pool = os.path.join(
        os.path.dirname(__file__), 'data', 'resource_pool.yaml')
    communication_DSL = "ring"

    # Test wrong deployment_setting input
    with pytest.raises(SuperscalerError):
        sc.init(apply_gradient_op, loss, None, strategy,
                communication_DSL, resource_pool)
    # Test wrong communication_DSL input
    with pytest.raises(SuperscalerError):
        sc.init(apply_gradient_op, loss, deployment_setting, strategy,
                None, resource_pool)
    # Test wrong resource_pool input
    with pytest.raises(SuperscalerError):
        sc.init(apply_gradient_op, loss, deployment_setting, strategy,
                communication_DSL, None)

    sc.init(apply_gradient_op, loss, deployment_setting, strategy,
            communication_DSL, resource_pool)

    # Check whether cache_dir is created
    assert(cache_dir == sc.get_cache_dir())
    if not os.path.exists(cache_dir):
        raise OSError
    # Check whether working_dir is sub_folder of cache_dir
    assert(os.path.samefile(cache_dir, os.path.dirname(sc.get_working_dir())))
    if not os.path.exists(sc.get_working_dir()):
        raise OSError

    # Test for is_initialized() function, Returns True after initialization
    assert(sc.is_initialized() is True)

    # Test for sc.run() after initialization is complete
    sc.run()

    # Clean tmp dir
    shutil.rmtree(cache_dir)
