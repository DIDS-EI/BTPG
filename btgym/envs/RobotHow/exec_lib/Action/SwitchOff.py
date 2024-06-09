from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small

class SwitchOff(RHAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RHAction.HAS_SWITCH
    valid_args_small = VHTAction_small.HAS_SWITCH

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsLeftHandEmpty(self)",f"IsNear(self,{arg[0]})",f"IsSwitchedOn({arg[0]})"} # IsLeftHandEmpty()至少有一只手是空闲的
        info["add"]={f"IsSwitchedOff({arg[0]})"}
        info["del_set"] = {f"IsSwitchedOn({arg[0]})"}
        info["cost"] = 8
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]