from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
import itertools
from btgym.envs.VirtualHome.exec_lib.Action.Put import Put


class RightPut(Put):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(VHAction.Objects, VHAction.SurfacePlaces))

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]
        self.target_place = self.args[1]

    @property
    def action_class_name(self):
        return Put.__name__
    @classmethod
    def get_info(cls,*arg):
        info = {}
        if arg[0] != 'Anything':
            info["pre"] = {f'IsRightHolding(self,{arg[0]})',f'IsNear(self,{arg[1]})'}
            info["add"] = {f'IsRightHandEmpty(self)',f'IsOn({arg[0]},{arg[1]})'}
            info["del_set"] = {f'IsRightHolding(self,{arg[0]})'}
            info["cost"] = 6
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

