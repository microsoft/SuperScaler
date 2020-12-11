from enum import Enum, unique


@unique
class TransNodeType(Enum):
    """
    Enum for communication primitives
    """
    SEND = 'send'
    RECV = 'recv'


@unique
class CompNodeType(Enum):
    """
    Enum for computation (data segment) operations
    """
    ADD = 'add'
    SUB = 'sub'
    MUL = 'mul'
    DIV = 'div'
    COPY = 'copy'
    CREATE = 'create'
