from btpg.envs.robowaiter.exec_lib._base.rw_action import RWAction
from btpg.behavior_tree import Status

class Make(RWAction):
    can_be_expanded = True
    num_args = 1
    valid_args = (
        "Coffee","Water","Dessert"
    )

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]
        self.op_type = 1
        if self.target_obj==self.valid_args[0]:
            self.op_type = 1
        elif self.target_obj==self.valid_args[1]:
            self.op_type = 2
        elif self.target_obj==self.valid_args[2]:
            self.op_type = 3

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

    def _update(self) -> Status:
        if self.scene.show_ui:
            self.scene.get_obstacle_point(self.scene.db, self.status, map_ratio=self.scene.map_ratio)

        self.scene.move_task_area(self.op_type)
        self.scene.op_task_execute(self.op_type)
        if self.scene.show_ui:
            self.scene.get_obstacle_point(self.scene.db, self.status, map_ratio=self.scene.map_ratio,update_info_count=1)

        return Status.RUNNING