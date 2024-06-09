from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
import itertools
from btgym.envs.RobotHow_Small.exec_lib.Action.Grab import Grab
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small


class LeftGrabFrom(Grab):
    can_be_expanded = False
    num_args = 2
    valid_args = list(itertools.product(RHSAction.GRABBABLE, RHSAction.CONTAINERS))
    valid_args_small = list(itertools.product(VHTAction_small.GRABBABLE, VHTAction_small.CONTAINERS))

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @property
    def action_class_name(self):
        return Grab.__name__

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"] = {"IsLeftHandEmpty(self)", f"IsIn({arg[0]},{arg[1]})", f"IsNear(self,{arg[1]})"}  # 至少有一只手是空闲的

        # 能打开就需要先打开
        if arg[1] in RHSAction.CAN_OPEN:
            info["pre"] |= {f"IsOpen({arg[1]})"}

        info["add"] = {f"IsLeftHolding(self,{arg[0]})", "IsLeftHandFull(self)"}
        info["del_set"] = {f"IsLeftHandEmpty(self)"}
        info["del_set"] |= {f'IsIn({arg[0]},{place})' for place in cls.CONTAINERS}
        info["cost"] = 5
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
