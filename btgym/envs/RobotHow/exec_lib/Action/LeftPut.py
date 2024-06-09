from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
import itertools
from btgym.envs.RobotHow.exec_lib.Action.Put import Put
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small

class LeftPut(Put):
    can_be_expanded = True
    num_args = 2

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
            info["pre"] = {f'IsLeftHolding(self,{arg[0]})',f'IsNear(self,{arg[1]})'}
            info["add"] = {f'IsLeftHandEmpty(self)',f'IsOn({arg[0]},{arg[1]})'}
            info["del_set"] = {f'IsLeftHolding(self,{arg[0]})'}

            if arg[0] in RHAction.cleaning_tools:
                info["del_set"] = {f'IsHoldingCleaningTool(self)'}

            info["cost"] = 6
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

