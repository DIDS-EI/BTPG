from btpg.envs.robothow.exec_lib._base.rh_action import RHAction
# from btpg.envs.robothow.exec_lib._base.VHTAction_small import VHTAction_small


class PlugOut(RHAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RHAction.HAS_PLUG
    # valid_args_small = VHTAction_small.HAS_PLUG

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsLeftHandEmpty(self)",f"IsNear(self,{arg[0]})",f"IsPlugged({arg[0]})"} # IsLeftHandEmpty()至少有一只手是空闲的
        info["add"]={f"IsUnplugged({arg[0]})"}
        info["del_set"] = {f"IsPlugged({arg[0]})"}

        if arg[0] in RHAction.HAS_PLUG:
            info["pre"] |= {f"IsSwitchedOff({arg[0]})"} # 拔电器插头时需要确保电器已经关闭

        info["cost"] = 8
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")