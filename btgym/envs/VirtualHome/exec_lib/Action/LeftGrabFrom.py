from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
import itertools
from btgym.envs.VirtualHome.exec_lib.Action.Grab import Grab

class LeftGrabFrom(Grab):
    can_be_expanded = False
    num_args = 2
    valid_args = list(itertools.product(VHAction.Objects, VHAction.CanOpenPlaces))

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @property
    def action_class_name(self):
        return Grab.__name__

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsLeftHandEmpty(self)",f"IsIn({arg[0]},{arg[1]})",f"IsNear(self,{arg[1]})",f"IsOpen({arg[1]})"} # 至少有一只手是空闲的
        info["add"]={f"IsLeftHolding(self,{arg[0]})","IsLeftHandFull(self)"}
        info["del_set"] = {f"IsLeftHandEmpty(self)"}
        info["del_set"] |= {f'IsIn({arg[0]},{place})' for place in cls.CanPutInPlaces}
        info["cost"] = 5
        return info



    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
