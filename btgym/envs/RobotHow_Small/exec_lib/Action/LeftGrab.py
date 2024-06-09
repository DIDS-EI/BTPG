from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RobotHow_Small.exec_lib.Action.Grab import Grab
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small

class LeftGrab(Grab):
    can_be_expanded = True
    num_args = 1

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @property
    def action_class_name(self):
        return Grab.__name__

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsLeftHandEmpty(self)",f"IsNear(self,{arg[0]})"} # 至少有一只手是空闲的
        info["add"]={f"IsLeftHolding(self,{arg[0]})","IsLeftHandFull(self)"}

        if arg[0] in cls.cleaning_tools:
            info["add"]|= {f"IsHoldingCleaningTool(self)"}
        if arg[0] in cls.cutting_tools:
            info["add"]|= {f"IsHoldingKnife(self)"}

        info["del_set"] = {f"IsLeftHandEmpty(self)"}
        info["del_set"] |= {f'IsOn({arg[0]},{place})' for place in cls.SURFACES}
        info["del_set"] |= {f'IsIn({arg[0]},{place})' for place in cls.CONTAINERS}
        info["cost"] = 5
        return info



    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
