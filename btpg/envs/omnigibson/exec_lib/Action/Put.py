from btpg.envs.omnigibson.exec_lib._base.og_action import OGAction
# from btpg.envs.robothow.exec_lib._base.VHTAction_small import VHTAction_small

import itertools

class Put(OGAction):
    can_be_expanded = False
    num_args = 2
    # valid_args = list(itertools.product(OGAction.GRABBABLE, OGAction.SURFACES))

    set_1_food = OGAction.GRABBABLE & (OGAction.EATABLE|OGAction.DRINKABLE|{"apple","bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
                 'milk','wine',"cereal","lime","salmon", "peach","pear","plum"})


    valid_args = set(list(itertools.product(OGAction.GRABBABLE-set_1_food, OGAction.SURFACES-{"towelrack","plate","fryingpan"})) \
                    + list(itertools.product(OGAction.GRABBABLE & {'towel'}, {"towelrack"})) \
                    + list(itertools.product(set_1_food, OGAction.SURFACES-{"towelrack","bathroomcounter"})))
    valid_args = list(valid_args)


    # set_1_food_small = VHTAction_small.GRABBABLE & (VHTAction_small.EATABLE|VHTAction_small.DRINKABLE|{"bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
    #              'milk','wine',"cereal"})
    # valid_args_small = set(list(itertools.product(VHTAction_small.GRABBABLE-set_1_food_small, VHTAction_small.SURFACES-{"towelrack","plate","fryingpan"})) \
    #                 + list(itertools.product(VHTAction_small.GRABBABLE & {'towel'}, {"towelrack"})) \
    #                 + list(itertools.product(set_1_food_small, VHTAction_small.SURFACES-{"towelrack","bathroomcounter"})))
    # valid_args_small = list(valid_args_small)


    def __init__(self, *args):
        super().__init__(*args)
