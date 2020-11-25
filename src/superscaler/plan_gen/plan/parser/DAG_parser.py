# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class DAGParser():
    def __init__(self, parser_type):
        self.type = parser_type

    def get_parser_type(self):
        return self.type
