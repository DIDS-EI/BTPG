from btgym.envs.RoboWaiter.exec_lib._base.RWCondition import RWCondition
from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
class On(RWCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = [tuple(RWAction.all_object),
            tuple(RWAction.tables_for_placement)]

