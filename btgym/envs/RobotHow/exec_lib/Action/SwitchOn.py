from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small

class SwitchOn(RHAction):
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
        info["pre"]={"IsLeftHandEmpty(self)",f"IsNear(self,{arg[0]})",f"IsSwitchedOff({arg[0]})"} # IsLeftHandEmpty()至少有一只手是空闲的
        info["add"]={f"IsSwitchedOn({arg[0]})"}
        info["del_set"] = {f"IsSwitchedOff({arg[0]})"}

        if arg[0] in RHAction.CAN_OPEN:
            info["pre"] |= {f"IsClose({arg[0]})"}

        if arg[0] in RHAction.HAS_PLUG:
            info["pre"] |= {f"IsPlugged({arg[0]})"}

        info["cost"] = 8
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")