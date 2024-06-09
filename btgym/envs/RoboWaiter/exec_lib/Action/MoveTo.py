from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction

class MoveTo(RWAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RWAction.all_object | RWAction.tables_for_placement | RWAction.tables_for_guiding

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info['pre'] = set()
        if arg[0] in RWAction.all_object:
            info['pre'] |= {f'Exists({arg[0]})'}

        info["add"] = {f'RobotNear({arg[0]})'}
        info["del_set"] = {f'RobotNear({place})' for place in cls.valid_args if place != arg[0]}

        info['cost'] = 15
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")