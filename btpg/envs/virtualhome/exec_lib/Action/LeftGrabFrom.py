from btpg.envs.virtualhome.exec_lib._base.vh_action import VHAction
import itertools
from btpg.envs.virtualhome.exec_lib.Action.Grab import Grab

class LeftGrabFrom(Grab):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(
        VHAction.Objects, sorted(VHAction.CanOpenPlaces | VHAction.SurfacePlaces)))

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @property
    def action_class_name(self):
        return Grab.__name__

    @property
    def script_args(self):
        # VirtualHome's GRAB takes a single object; the place only refines
        # the planning model.
        return self.args[:1]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        if arg[1] in cls.CanOpenPlaces:
            info["pre"]={"IsLeftHandEmpty(self)",f"IsIn({arg[0]},{arg[1]})",f"IsNear(self,{arg[1]})",f"IsOpen({arg[1]})"}
        else:
            info["pre"]={"IsLeftHandEmpty(self)",f"IsOn({arg[0]},{arg[1]})",f"IsNear(self,{arg[1]})"}
        info["add"]={f"IsLeftHolding(self,{arg[0]})"}
        info["del_set"] = {f"IsLeftHandEmpty(self)"}
        info["del_set"] |= {f'IsIn({arg[0]},{place})' for place in cls.CanPutInPlaces}
        info["del_set"] |= {f'IsOn({arg[0]},{place})' for place in cls.SurfacePlaces}
        info["cost"] = 5
        return info



    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
