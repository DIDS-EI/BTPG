from btpg.envs.virtualhome.exec_lib._base.vh_action import VHAction
from btpg.envs.virtualhome.exec_lib.Action.PutIn import PutIn

class LeftPutIn(PutIn):
    can_be_expanded = True
    num_args = 2


    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]
        self.target_place = self.args[1]

    @property
    def action_class_name(self):
        return PutIn.__name__

    @classmethod
    def get_info(cls,*arg):
        info = {}
        if arg[0] != 'Anything':
            info["pre"] = {f'IsLeftHolding(self,{arg[0]})',f"IsNear(self,{arg[1]})",f"IsOpen({arg[1]})"}
            info["add"] = {f'IsLeftHandEmpty(self)',f'IsIn({arg[0]},{arg[1]})'}
            info["del_set"] = {f'IsLeftHolding(self,{arg[0]})'}
            info["cost"] = 10
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

