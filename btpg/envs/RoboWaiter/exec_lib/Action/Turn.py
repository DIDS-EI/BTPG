from btpg.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
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

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"] = set()
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