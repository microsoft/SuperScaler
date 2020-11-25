# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc


class Resource(abc.ABC):
    @abc.abstractmethod
    def get_name(self):
        pass

    def __eq__(self, obj):
        return self.get_name() == obj.get_name()

    def __hash__(self):
        return id(self)
