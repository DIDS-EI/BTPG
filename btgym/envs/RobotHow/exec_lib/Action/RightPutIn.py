from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow.exec_lib.Action.PutIn import PutIn

class RightPutIn(PutIn):
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
            info["pre"] = {f'IsRightHolding(self,{arg[0]})',f"IsNear(self,{arg[1]})"}

            # puin 之前要插上电？
            if arg[1] in RHAction.HAS_PLUG:
                info["pre"] |= {f"IsPlugged({arg[1]})"}

            # 能打开就需要先打开
            if arg[1] in RHAction.CAN_OPEN:
                info["pre"] |= {f"IsOpen({arg[1]})"}

            info["add"] = {f'IsRightHandEmpty(self)',f'IsIn({arg[0]},{arg[1]})'}
            info["del_set"] = {f'IsRightHolding(self,{arg[0]})'}
            info["cost"] = 10

            if arg[0] in RHAction.cleaning_tools:
                info["del_set"] = {f'IsHoldingCleaningTool(self)'}
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

