from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small

class Open(RHAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RHAction.CAN_OPEN
    valid_args_small = VHTAction_small.CAN_OPEN

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"IsClose({arg[0]})",f"IsNear(self,{arg[0]})","IsLeftHandEmpty(self)"} # IsLeftHandEmpty()至少有一只手是空闲的

        if arg[0] in RHAction.HAS_SWITCH:
            info["pre"] |= {f"IsSwitchedOff({arg[0]})"}

        info["add"]={f"IsOpen({arg[0]})"}
        info["del_set"] = {f"IsClose({arg[0]})"}
        info["cost"] = 3
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")