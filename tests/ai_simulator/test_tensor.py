# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from superscaler.ai_simulator.simulator.tensor import Tensor, TensorException


def test_tensor_module():
    # Test tensor initialization failure
    # Illegal tensor type
    with pytest.raises(TensorException):
        tensor = Tensor("long long", 18)
    with pytest.raises(TensorException):
        tensor = Tensor(-1, 18)
    with pytest.raises(TensorException):
        tensor = Tensor("DT_VARIANT", 18)

    # Illegal tensor size
    with pytest.raises(TensorException):
        tensor = Tensor("DT_FLOAT", -18)
    with pytest.raises(TensorException):
        tensor = Tensor("DT_FLOAT", 1.5)

    tensor = Tensor("DT_DOUBLE", 10)
    assert tensor.get_bytes_size() == 8*10

    # Test check_tensor_type
    assert Tensor.check_tensor_type("DT_FLOAT")
    assert not Tensor.check_tensor_type("DT_RESOURCE")
    assert not Tensor.check_tensor_type("NOT_A_VALID_TYPE")
