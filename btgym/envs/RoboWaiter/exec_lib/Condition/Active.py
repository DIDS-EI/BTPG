from btgym.envs.RoboWaiter.exec_lib._base.RWCondition import RWCondition

class Activate(RWCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = {'AC','TubeLight','HallLight'}

