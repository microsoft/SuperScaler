import pytest
from simulator.tensor import *

def test_tensor_module():
    # Test tensor initialization failure
    # Illegal tensor type
    with pytest.raises(TensorException):
        tensor = Tensor('long long', 18)
    
    # Illegal tensor size
    with pytest.raises(TensorException):
        tensor = Tensor('float', -18)
    with pytest.raises(TensorException):
        tensor = Tensor('float', 1.5)
    
    tensor = Tensor('double',10)
    assert tensor.get_bytes_size()==8*10