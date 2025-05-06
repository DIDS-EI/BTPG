from btpg.envs.robothow.exec_lib._base.rh_action import RHAction
# from btpg.envs.robothow.exec_lib._base.VHTAction_small import VHTAction_small

class Grab(RHAction):
    can_be_expanded = False
    num_args = 1
    valid_args = RHAction.GRABBABLE
    # valid_args_small = VHTAction_small.GRABBABLE

    def __init__(self, *args):
        super().__init__(*args)
