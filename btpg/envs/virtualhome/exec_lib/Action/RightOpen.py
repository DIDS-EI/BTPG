from btpg.envs.virtualhome.exec_lib._base.vh_action import VHAction
from btpg.envs.virtualhome.exec_lib.Action.Open import Open

class RightOpen(Open):
    can_be_expanded = True
    num_args = 1
    valid_args = VHAction.CanOpenPlaces

    def __init__(self, *args):
        super().__init__(*args)

    @property
    def action_class_name(self):
        return Open.__name__

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"IsClose({arg[0]})",f"IsNear(self,{arg[0]})","IsRightHandEmpty(self)"}
        info["add"]={f"IsOpen({arg[0]})"}
        info["del_set"] = {f"IsClose({arg[0]})"}
        info["cost"] = 3
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
