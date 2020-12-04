# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class CommDSLSpecError(Exception):

    def __init__(self, message='CommDSL spec error occured.'):
        super().__init__(type(self).__name__ + ': ' + message)


class CommDSLRuntimeError(Exception):

    def __init__(self, message="CommDSL runtime error occured."):
        super().__init__(type(self).__name__ + ': ' + message)


class CommDSLCodeGenError(Exception):

    def __init__(self, message="CommDSL codegen error occured."):
        super().__init__(type(self).__name__ + ': ' + message)
