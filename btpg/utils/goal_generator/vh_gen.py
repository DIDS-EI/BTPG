import random
from btpg.utils.goal_generator.goal_gen_base import GoalGenerator


class VirtualHomeGoalGen(GoalGenerator):

    def __init__(self):
        super().__init__()
        self.SURFACES = {"kitchentable","plate","nightstand","desk","cabinet","bathroomcounter","stove"} # put
        self.SittablePlaces =  {"bed","sofa","chair","Bench"}  # sit
        self.CAN_OPEN= {"fridge","dishwasher","microwave","stove","cabinet"}  # open
        self.CONTAINERS={"fridge","dishwasher","microwave","stove","cabinet"}  # put in
        self.GRABBABLE={"bananas",'chicken', 'cutlets','breadslice','chips','chocolatesyrup',
                 'cupcake','milk','juice','wine',
                 'cutleryknife','fryingpan','dishbowl','plate',
                 'book',"waterglass"
                 }  # grab
        self.HAS_SWITCH = {"tv","faucet","lightswitch","dishwasher","coffeemaker","toaster","microwave",
                            "tablelamp","computer"}  # switch on #candle  cellphone wallphone washingmachine不行# faucet 浴室龙头


        self.AllObject = self.SURFACES | self.SittablePlaces | self.CAN_OPEN | self.CONTAINERS | self.GRABBABLE |\
                     self.HAS_SWITCH

        self.cond_pred = {'IsOn_', 'IsIn_', 'IsOpen_', 'IsSwitchedOn_', 'IsNear_self_'}

    def condition2goal(self,condition,diffcult_type="multi"):
        goal = ''
        if condition == 'IsOn_':
            A = random.choice(list(self.GRABBABLE))
            B = random.choice(list(self.SURFACES))
            goal = 'IsOn_' + A + '_' + B
        elif condition == 'IsIn_':
            A = random.choice(list(self.GRABBABLE))
            B = random.choice(list(self.CONTAINERS))
            A = A.split('-')[0]
            B = B.split('-')[0]
            goal += 'IsIn_' + A + '_' + B
            if diffcult_type!="single":
                if B in self.CAN_OPEN:
                    goal += ' & IsClose_' + B
        elif condition == 'IsOpen_':
            goal = 'IsOpen_' + random.choice(list(self.CAN_OPEN))
        elif condition == 'IsClose_':
            goal = 'IsClose_' + random.choice(list(self.CAN_OPEN))
        elif condition == 'IsSwitchedOn_':
            A = random.choice(list(self.HAS_SWITCH))
            goal += 'IsSwitchedOn_' + A
        elif condition == 'IsSwitchedOff_':
            goal += 'IsSwitchedOff_' + random.choice(list(self.HAS_SWITCH))
        elif condition == 'IsNear_self_':
            goal = 'IsNear_self_' + random.choice(list(self.AllObject))
        return goal


    def get_goals_string(self,diffcult_type="multi"):
        goal_list = []
        if diffcult_type == "single":
            goal_mount = random.randint(1, 1)
        elif diffcult_type == "multi":
            goal_mount = random.randint(2, 3)
        elif diffcult_type == "mix":
            goal_mount = random.randint(1, 3)

        conditions = []
        for i in range(goal_mount):
            condition = random.choice(list(self.cond_pred))
            conditions.append(condition)
        for condition in conditions:
            goal = self.condition2goal(condition,diffcult_type=diffcult_type)

            # Keep no more than three.
            if '&' in goal:
                split_goals = goal.split(' & ')
                goal_list.extend(split_goals)
            else:
                goal_list.append(goal)

            if len(goal_list)>=3:
                goal_list=goal_list[:3]

        goal_string = ' & '.join(goal_list)
        return goal_string