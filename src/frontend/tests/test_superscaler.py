import os
import pytest
import shutil
from superscaler import Superscaler, SuperscalerError


def test_superscaler():

    # Init SuperscalerBasics class
    sc = Superscaler()

    # Test for set_cache_dir function
    # None path test
    assert(sc.set_cache_dir(None) is False)
    # Int path test
    assert(sc.set_cache_dir(1) is False)
    cache_dir = os.path.join(os.path.dirname(__file__), 'tmp/')
    print(cache_dir)
    assert(sc.set_cache_dir(cache_dir) is True)

    # Test for error by calling sc.run() before initialization is complete
    with pytest.raises(SuperscalerError):
        sc.run()

    # Test for is_initialized() function, default setting is false
    assert(sc.is_initialized() is False)

    user_model = "TODO user_model"
    deployment_plan = "TODO deployment_plan"
    strategy = "TODO strategy"
    resource_pool = "resource_pool"
    sc.init(user_model, deployment_plan, strategy, resource_pool)

    # Check whether cache_dir is created
    if not os.path.exists(cache_dir):
        raise OSError

    # Test for is_initialized() function, Returns True after initialization
    assert(sc.is_initialized() is True)

    # Test for sc.run() after initialization is complete
    sc.run()

    # Clean tmp dir
    shutil.rmtree(cache_dir)
