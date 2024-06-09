from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction

class Clean(RWAction):
    can_be_expanded = True
    num_args = 1
    valid_args = {
        'Table1','Floor','Chairs'
    }

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]= {f'Holding(Nothing)'}
        info["add"] = {f'IsClean({arg[0]})'}
        info["del_set"] = set()
        info['cost'] = 10
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")