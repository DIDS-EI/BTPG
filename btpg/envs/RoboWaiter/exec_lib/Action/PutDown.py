from btpg.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btpg.behavior_tree import Status
import itertools
class PutDown(RWAction):
    can_be_expanded = True
    num_args = 2
    valid_args = list(itertools.product(RWAction.all_object, RWAction.tables_for_placement))
    valid_args.append(('Anything','Anywhere'))
    valid_args = set(valid_args)
    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]
        self.target_place = self.args[1]

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

    def _update(self) -> Status:
        # self.scene.test_move()

        if self.target_obj != 'Anything':
            op_type = 17
            release_pos = list(RWAction.place_xyz_dic[self.target_place])
            # # 原始吧台处:[247.0, 520.0, 100.0], 空调开关旁吧台:[240.0, 40.0, 70.0], 水杯桌:[-70.0, 500.0, 107]
            # # 桌子2:[-55.0, 0.0, 107],桌子3:[-55.0, 150.0, 107]
            if RWAction.num_of_obj_on_place[self.target_place] >= 1:
                release_pos[1] += 25

            RWAction.num_of_obj_on_place[self.target_place] += 1

            self.scene.move_task_area(op_type, release_pos=release_pos)

            if self.target_obj == "Chips":
                release_pos[2] += 3
            self.scene.op_task_execute(op_type, release_pos=release_pos)

        if self.scene.show_ui:
            self.scene.get_obstacle_point(self.scene.db, self.status, map_ratio=self.scene.map_ratio,
                                          update_info_count=1)

        self.scene.state["condition_set"] |= (self.info["add"])
        self.scene.state["condition_set"] -= self.info["del_set"]

        return Status.RUNNING