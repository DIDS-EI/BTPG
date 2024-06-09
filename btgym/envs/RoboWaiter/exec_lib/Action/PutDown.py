from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
import itertools
class PutDown(RWAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(RWAction.all_object, RWAction.tables_for_placement))
    valid_args.append(('Anything','Anywhere'))
    valid_args = set(valid_args)
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        if arg[0] != 'Anything':
            info = {}
            info["pre"] = {f'Holding({arg[0]})',f'RobotNear({arg[1]})'}
            info["add"] = {f'On({arg[0]},{arg[1]})'}
            info["del_set"] = {f'Holding({arg[0]})'}
            info['cost'] = 20 #1000
        else:
            info = {}
            info["pre"] = set()
            info['add'] = {f'Holding(Nothing)'}
            info['del_set'] = {f'Holding({obj})' for obj in cls.all_object}
            info['cost'] = 8

        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")