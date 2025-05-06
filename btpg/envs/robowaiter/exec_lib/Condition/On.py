from btpg.envs.robowaiter.exec_lib._base.rw_condition import RWCondition
from btpg.envs.robowaiter.exec_lib._base.rw_action import RWAction
class On(RWCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = [tuple(RWAction.all_object),
            tuple(RWAction.tables_for_placement)]

