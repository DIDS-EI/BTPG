from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RobotHow_Small.exec_lib.Action.Grab import Grab

class RightGrab(Grab):
    can_be_expanded = True
    num_args = 1
    # obj1 is reachable (not inside some closed container)

    def __init__(self, *args):
        super().__init__(*args)

    @property
    def action_class_name(self):
        # 根据需要，这里可以返回当前类名或父类名
        # 例如，直接返回父类的名字
        return Grab.__name__

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsRightHandEmpty(self)",f"IsNear(self,{arg[0]})"} # 至少有一只手是空闲的
        info["add"]={f"IsRightHolding(self,{arg[0]})","IsRightHandFull(self)"}

        if arg[0] in cls.cleaning_tools:
            info["add"]|= {f"IsHoldingCleaningTool(self)"}
        if arg[0] in cls.cutting_tools:
            info["add"]|= {f"IsHoldingKnife(self)"}

        info["del_set"] = {f"IsRightHandEmpty(self)"}
        info["del_set"] |= {f'IsOn({arg[0]},{place})' for place in cls.SURFACES}
        info["del_set"] |= {f'IsIn({arg[0]},{place})' for place in cls.CONTAINERS}
        info["cost"] = 5
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
