# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class TensorException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def print_error_info(self):
        print(self.error_info)


class Tensor():
    __tensor_type_xlate_tbl = {
        'DT_FLOAT': 4,
        'DT_DOUBLE': 8,
        'DT_INT32': 4,
        'DT_UINT8': 1,
        'DT_INT16': 2,
        'DT_INT8': 1,
        'DT_STRING': None,
        'DT_COMPLEX64': 8,    # Single-precision complex
        'DT_INT64': 8,
        'DT_BOOL': None,  # sizeof(bool) is not required to be 1 in C++.
        'DT_QINT8': 1,       # Quantized int8
        'DT_QUINT8': 1,      # Quantized uint8
        'DT_QINT32': 4,      # Quantized int32
        'DT_BFLOAT16': 2,    # Float32 truncated to 16 bits.
        'DT_QINT16': 2,      # Quantized int16
        'DT_QUINT16': 2,     # Quantized uint16
        'DT_UINT16': 2,
        'DT_COMPLEX128': 16,  # Double-precision complex
        'DT_HALF': 2,
        'DT_RESOURCE': None,
        'DT_VARIANT': None,     # Arbitrary C++ data types
        'DT_UINT32': 4,
        'DT_UINT64': 8
    }

    def __init__(self, tensor_type, tensor_size=0):
        '''Init a Tensor

        Args:
            tensor_type: the type of the tensor, e.g. DT_FLOAT
            tensor_size: the total number of data,

        Note: total_bytes = tensor_size * tensor_type_byte_size
        '''
        # Check the validity of inputs
        if tensor_type not in Tensor.__tensor_type_xlate_tbl \
                or Tensor.__tensor_type_xlate_tbl[tensor_type] is None:
            raise TensorException(
                '[ERROR] Tensor initialization failure because unsupported '
                + 'tensor_type: %s' % tensor_type)

        if not isinstance(tensor_size, int):
            raise TensorException(
                '[ERROR] Tensor initialization failure because data size is '
                + 'not an integer: %s' % tensor_size)
        if tensor_size < 0:
            raise TensorException(
                '[ERROR] Tensor initialization failure because data size is '
                + 'negative: %s' % tensor_size)

        # int, the data type stored in the node
        self.tensor_type = tensor_type
        # int, total_bytes = tensor_size * tensor_type_byte_size
        self.size = tensor_size

    def get_bytes_size(self):
        return Tensor.__tensor_type_xlate_tbl[self.tensor_type] * self.size

    @staticmethod
    def check_tensor_type(tensor_type):
        '''Return True is tensor_type is supported
        '''
        return tensor_type in Tensor.__tensor_type_xlate_tbl \
            and Tensor.__tensor_type_xlate_tbl[tensor_type] is not None
