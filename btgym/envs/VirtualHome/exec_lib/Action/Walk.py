from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
class Walk(VHAction):
    can_be_expanded = True
    num_args = 1
    # obj1 is reachable (not inside some closed container) or obj1 is a room.
    valid_args = VHAction.SurfacePlaces | VHAction.SittablePlaces | VHAction.Objects | \
                 VHAction.CanPutInPlaces | VHAction.HasSwitchObjects | VHAction.SittablePlaces
    # valid_args = VHAction.HasSwitchObjects


    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsStanding(self)"}
        info["add"]={f"IsNear(self,{arg[0]})"}
        info["del_set"] = {f'IsNear(self,{place})' for place in cls.valid_args if place != arg[0]}
        info["cost"] = 15
        return info

    def change_condition_set(self):
        # del_list = []
        # for c in self.agent.condition_set:
        #     if "IsNear" in c:
        #         del_list.append(c)
        # for c in del_list:
        #     self.agent.condition_set.remove(c)
        #
        # self.agent.condition_set.add(f"IsNear(self,{self.args[0]})")
        self.agent.condition_set |= (self.info["add"]) #self.agent.condition_set.update(self.info["add"])
        self.agent.condition_set -= self.info["del_set"] #self.agent.condition_set.difference_update(self.info["del_set"])
