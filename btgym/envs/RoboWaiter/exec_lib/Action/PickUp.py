from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction

class PickUp(RWAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RWAction.all_object
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"] = {f'RobotNear({arg[0]})','Holding(Nothing)'}
        info["add"] = {f'Holding({arg[0]})'}
        info["del_set"] = {f'Holding(Nothing)',f'Exists({arg[0]})'}
        for place in cls.all_place:
            info["del_set"] |= {f'On({arg[0]},{place})'}
        info['cost'] = 2
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")