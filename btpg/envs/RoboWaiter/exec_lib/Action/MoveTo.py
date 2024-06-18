from btpg.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btpg.behavior_tree import Status
class MoveTo(RWAction):
    can_be_expanded = True
    num_args = 1
    valid_args = RWAction.all_object | RWAction.tables_for_placement | RWAction.tables_for_guiding

    def __init__(self, *args):
        super().__init__(*args)
        self.target_place = args[0]

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

    def _update(self) -> Status:

        # Take a photo
        if self.scene.show_ui:
            self.scene.get_obstacle_point(self.scene.db, self.status, map_ratio=self.scene.map_ratio, is_nav=True)

        # #####################################
        # Move to a fixed location
        if self.target_place in RWAction.place_xy_yaw_dic:
            goal = RWAction.place_xy_yaw_dic[self.target_place]
            if self.scene.is_nav_walk:
                self.scene.navigator.navigate(goal=(goal[0] + 1, goal[1]), animation=False)
                self.scene.walk_to(goal[0]+2, goal[1], goal[2])
            else:
                self.scene.walk_to(goal[0]+1,goal[1],goal[2])
        # Move to the side of an item
        else:
            # Check if it is properly contained
            if self.target_place in RWAction.container_dic:
                target_name = RWAction.container_dic[self.target_place]
            else:
                target_name = self.target_place
            # Find the closest object of this type by name
            obj_id = -1
            min_dis = float('inf')
            obj_dict = self.scene.status.objects
            if len(obj_dict)!=0:
                # Get obj_id
                for id,obj in enumerate(obj_dict):
                    if obj.name == target_name:
                        obj_info = obj_dict[id]
                        dis = self.scene.cal_distance_to_robot(obj_info.location.X, obj_info.location.Y, obj_info.location.Z)
                        if dis<min_dis:
                            min_dis = dis
                            obj_id = id
            if obj_id == -1:
                return Status.FAILURE
            self.scene.move_to_obj(obj_id=obj_id)
            # #####################################

        if self.scene.show_ui:
            self.scene.get_obstacle_point(self.scene.db, self.status, map_ratio=self.scene.map_ratio, is_nav=True)

        return Status.RUNNING

