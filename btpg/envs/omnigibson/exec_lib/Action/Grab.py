from btpg.envs.omnigibson.exec_lib._base.og_action import OGAction
# from btpg.envs.robothow.exec_lib._base.VHTAction_small import VHTAction_small

class Grab(OGAction):
    can_be_expanded = False
    num_args = 1
    valid_args = OGAction.GRABBABLE
    # valid_args_small = VHTAction_small.GRABBABLE

    def __init__(self, *args):
        super().__init__(*args)
