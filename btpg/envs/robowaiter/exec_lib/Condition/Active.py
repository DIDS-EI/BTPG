from btpg.envs.robowaiter.exec_lib._base.rw_condition import RWCondition

class Active(RWCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = {'AC','TubeLight','HallLight'}

