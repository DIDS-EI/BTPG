from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction

class Make(RWAction):
    can_be_expanded = True
    num_args = 1
    valid_args = {
        "Coffee","Water","Dessert"
    }

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]= {f'Holding(Nothing)'}
        info['del_set'] = set()
        info['add'] = {f'Exists({arg[0]})'}
        if arg[0] == "Coffee":
            info["add"] |= {f'On({arg[0]},CoffeeStation)'}
        elif arg[0] == "Water":
            info["add"] |= {f'On({arg[0]},WaterStation)'}
        elif arg[0] == "Dessert":
            info["add"] |= {f'On({arg[0]},Bar)'}
        info['cost'] = 5
        return info


    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")