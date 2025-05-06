from btpg.envs.robowaiter.exec_lib._base.rw_action import RWAction
from btpg.behavior_tree import Status
class Clean(RWAction):
    can_be_expanded = True
    num_args = 1
    valid_args = {
        'Table1','Floor','Chairs'
    }

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]
        self.op_type = 5
        if self.target_obj=="Table1":
            self.op_type = 5
        elif self.target_obj=="Floor":
            self.op_type = 4
        elif self.target_obj=="Chairs":
            self.op_type = 7

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

    def _update(self) -> Status:

        self.scene.move_task_area(self.op_type)
        self.scene.op_task_execute(self.op_type)
        self.scene.get_obstacle_point(self.scene.db, self.status, map_ratio=self.scene.map_ratio)

        return Status.RUNNING