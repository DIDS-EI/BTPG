from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction

class Watch(RHAction):
    can_be_expanded = False
    num_args = 1
    # obj1 is reachable (not inside some closed container)
    valid_args = RHAction.AllObject

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={f"IsNear(self,{arg[0]})"}
        info["add"]={f"IsWatching(self,{arg[0]})"}
        info["del_set"] = {f'"IsWatching(self,{obj})' for obj in cls.valid_args if obj != arg[0]}
        info["cost"] = 2
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
        # self.agent.condition_set.add(f"IsWatching(self,{self.args[0]})")