class TensorException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def print_error_info(self):
        print(self.error_info)


'''
Tensor
Attribute name      Type            Description
type                String          Data type of node. e.g. float/int8
size                Int             Total number of data.
                                    Thus, total_bytes = size * sizeof(type)
'''


class Tensor():
    # Supported tensor type
    valid_type = {'float': 4, 'double': 8, 'char': 1,
                  'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
                  'complex64': 8, 'complex128': 16}

    def __init__(self,
                 tensor_type='int8',
                 tensor_size=0):
        # String, the data type stored in the node
        self.type = tensor_type
        # Check type
        if self.type not in Tensor.valid_type:
            raise TensorException(
                '[ERROR] Tensor initialization failure because unsupported '
                + 'data_type: %s' % self.type)
        # Int, size * sizeof(type) = data_total_bytes
        self.size = tensor_size
        # Check size
        if self.size < 0:
            raise TensorException(
                '[ERROR] Tensor initialization failure because data size is '
                + 'negative: %s' % self.size)
        if not isinstance(self.size, int):
            raise TensorException(
                '[ERROR] Tensor initialization failure because data size is '
                + 'not an integer: %s' % self.size)

    def get_bytes_size(self):
        return Tensor.valid_type[self.type]*self.size
