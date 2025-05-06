from btpg.envs.robowaiter.exec_lib._base.rw_action import RWAction
import itertools
from btpg.behavior_tree import Status
class Turn(RWAction):
    can_be_expanded = True
    num_args = 2
    valid_args = [('AC','TubeLight','HallLight','Curtain'),
            ('On','Off')]

    valid_args = list(itertools.product(valid_args[0], valid_args[1]))
    valid_args.extend([('ACTemperature','Up'),('ACTemperature','Down')])
    valid_args = tuple(valid_args)

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]
        self.op = self.args[1]
        self.op_type = 13

        if self.target_obj=="AC":
            self.op_type = 13
        elif self.target_obj=="ACTemperature":
            if self.op == 'Up':
                self.op_type = 14
            elif self.op == 'Down':
                self.op_type = 15
        elif self.target_obj=="TubeLight":
            if self.op == 'On':
                self.op_type = 6
            elif self.op == 'Off':
                self.op_type = 8
        elif self.target_obj=="HallLight":
            if self.op == 'On':
                self.op_type = 9
            elif self.op == 'Off':
                self.op_type = 10
        elif self.target_obj=="Curtain":
            if self.op == 'On':
                self.op_type = 12
            elif self.op == 'Off':
                self.op_type = 11
    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"] = set()
        info["del_set"] = set()
        if arg[0] == "TubeLight" or arg[0] == "HallLight" or arg[0] == 'AC':
            info["pre"] |= {f'Holding(Nothing)'}
            if arg[1] == "On":
                info["add"] = {f'Active({arg[0]})'}
            elif arg[1]=="Off":
                info["pre"] |= {f'Active({arg[0]})'}
                info["del_set"] = {f'Active({arg[0]})'}

        elif arg[0]=='ACTemperature':
            info["pre"] = {f'Holding(Nothing)',f'Active(AC)'}
            if arg[1]=="Up":
                info["del_set"] = {f'Low({arg[0]})'}
            elif arg[1]=="Down":
                info["add"] = {f'Low({arg[0]})'}

        elif arg[0]=='Curtain':
            if arg[1]=="On":
                info["pre"] |= {f'Closed({arg[0]})'}
                info["del_set"] = {f'Closed({arg[0]})'}
            elif arg[1]=="Off":
                info["add"] = {f'Closed({arg[0]})'}
        info['cost'] = 3
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
            self.scene.get_obstacle_point(self.scene.db, self.status, map_ratio=self.scene.map_ratio)

        return Status.RUNNING