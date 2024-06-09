import py_trees as ptree

from typing import Any
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.behavior_tree import Status

class StandUp(RHSAction):
    can_be_expanded = False
    num_args = 0
    valid_args = set()

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls):
        info = {}
        info["pre"]={"IsSitting(self)"}
        info["add"]={f"IsStanding(self)"}
        info["del_set"] = {f"IsSitting(self)"}
        info["del_set"] |= {f'IsSittingOn(self,{place})' for place in cls.SITTABLE}
        info["cost"] = 15
        return info
    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]