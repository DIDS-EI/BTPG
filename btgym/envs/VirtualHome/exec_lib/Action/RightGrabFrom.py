from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
import itertools
from btgym.envs.VirtualHome.exec_lib.Action.Grab import Grab

class RightGrabFrom(Grab):
    can_be_expanded = False
    num_args = 2
    # obj1 is reachable (not inside some closed container)
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
        info["pre"]={"IsRightHandEmpty(self)",f"IsIn({arg[0]},{arg[1]})",f"IsNear(self,{arg[1]})",f"IsOpen({arg[1]})"} # 至少有一只手是空闲的
        info["add"]={f"IsRightHolding(self,{arg[0]})","IsRightHandFull(self)"}
        info["del_set"] = {f"IsRightHandEmpty(self)"}
        info["del_set"] |= {f'IsIn({arg[0]},{place})' for place in cls.CanPutInPlaces}
        info["cost"] = 5
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
