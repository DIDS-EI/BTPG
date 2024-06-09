from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small

class Wipe(RHAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RHAction.AllObject - RHAction.EATABLE - RHAction.DRINKABLE-RHAction.WASHABLE
    valid_args_small = VHTAction_small.AllObject - VHTAction_small.EATABLE - VHTAction_small.DRINKABLE-RHAction.WASHABLE

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"IsHoldingCleaningTool(self)",f"IsNear(self,{arg[0]})"} # IsLeftHandEmpty()至少有一只手是空闲的

        # if arg[0] in RHAction.things_need_rag:
        #     info["pre"] |= {f"IsRightHolding(self,rag)"}
        # elif arg[0] in RHAction.things_need_duster:
        #     info["pre"] |= {f"IsRightHolding(self,duster)"}
        # elif arg[0] in RHAction.things_need_brush:
        #     info["pre"] |= {f"IsRightHolding(self,brush)"}
        # elif arg[0] in RHAction.things_need_papertowel:
        #     info["pre"] |= {f"IsRightHolding(self,papertowel)"}

        info["add"]={f"IsClean({arg[0]})"}
        info["del_set"] = set()
        info["cost"] = 9
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")