import py_trees as ptree
from typing import Any
import enum
from py_trees.common import Status


# base_nodes Behavior
class BahaviorNode(ptree.behaviour.Behaviour):
    is_composite = False
    can_be_expanded = False
    num_params = 0
    valid_params='''
        None
        '''
    agent = None
    env = None
    print_name_prefix = ""


    @classmethod
    def get_ins_name(cls,*args):
        name = cls.__name__
        if len(args) > 0:
            ins_name = f'{name}({",".join(list(args))})'
        else:
            ins_name = f'{name}()'
        return ins_name

    def __init__(self,*args):
        ins_name = self.__class__.get_ins_name(*args)
        self.args = args

        super().__init__(ins_name)


    def update(self) -> Status:
        print("this is just a base_nodes behavior node.")
        return Status.INVALID


    def setup(self, **kwargs: Any) -> None:
        return super().setup(**kwargs)

    def initialise(self) -> None:
        return super().initialise()

    def terminate(self, new_status: Status) -> None:
        return super().terminate(new_status)

    @property
    def print_name(self):
        return f'{self.print_name_prefix}{self.get_ins_name(*self.args)}'

    @property
    def arg_str(self):
        return ",".join(self.args)