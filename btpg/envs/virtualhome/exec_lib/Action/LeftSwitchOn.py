from btpg.envs.virtualhome.exec_lib._base.vh_action import VHAction
from btpg.envs.virtualhome.exec_lib.Action.SwitchOn import SwitchOn

class LeftSwitchOn(SwitchOn):
    can_be_expanded = True
    num_args = 1
    valid_args = VHAction.HasSwitchObjects

    def __init__(self, *args):
        super().__init__(*args)

    @property
    def action_class_name(self):
        return SwitchOn.__name__

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsLeftHandEmpty(self)",f"IsNear(self,{arg[0]})",f"IsSwitchedOff({arg[0]})"}
        if arg[0] in cls.CAN_OPEN:
            info["pre"]|={f"IsClose({arg[0]})"}
        info["add"]={f"IsSwitchedOn({arg[0]})"}
        info["del_set"] = {f"IsSwitchedOff({arg[0]})"}
        info["cost"] = 8
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
