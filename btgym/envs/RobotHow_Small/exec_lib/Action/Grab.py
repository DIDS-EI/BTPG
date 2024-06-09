from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small

class Grab(RHSAction):
    can_be_expanded = False
    num_args = 1
    valid_args = RHSAction.GRABBABLE
    valid_args_small = VHTAction_small.GRABBABLE

    def __init__(self, *args):
        super().__init__(*args)
