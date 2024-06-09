from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction

class SwitchOn(VHAction):
    can_be_expanded = True
    num_args = 1
    valid_args = VHAction.HasSwitchObjects


    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsLeftHandEmpty(self)",f"IsNear(self,{arg[0]})",f"IsSwitchedOff({arg[0]})"} # IsLeftHandEmpty()至少有一只手是空闲的
        info["add"]={f"IsSwitchedOn({arg[0]})"}
        info["del_set"] = {f"IsSwitchedOff({arg[0]})"}
        info["cost"] = 8
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")