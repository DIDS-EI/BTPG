

import random

class GoalGenerator:

    def __int__(self):
        self.SURFACES = {"kitchencabinet", "bed"}
        self.SITTABLE = {"bed"}
        self.CAN_OPEN = {"fridge", "window", "microwave", "kitchencabinet"}
        self.CONTAINERS = {"fridge", "garbagecan", "microwave", "kitchencabinet"}
        self.GRABBABLE = {"apple", 'wine', 'plate', "rag", "kitchenknife", "cutlets"}
        self.cleaning_tools = {"rag"}
        self.cutting_tools = {"kitchenknife"}
        self.HAS_SWITCH = {"tv", "faucet", "candle", "microwave"}
        self.HAS_PLUG = {"tv", "mouse", "fridge", "microwave"}
        self.CUTABLE = {"apple", "cutlets"}
        self.WASHABLE = {"apple", "rag", "kitchenknife", "cutlets"}
        self.EATABLE = {"apple", 'cutlets'}
        self.DRINKABLE = {'wine'}

        self.AllObject = self.SURFACES | self.SITTABLE | self.CAN_OPEN | self.CONTAINERS | self.GRABBABLE | \
                    self.HAS_SWITCH | self.CUTABLE | self.EATABLE | self.DRINKABLE

        self.cond_pred = {'IsOn_', 'IsIn_', 'IsOpen_', 'IsSwitchedOn_', 'IsClean_',
                     'IsPlugged_', 'IsCut_', 'IsNear_self_'}





    def condition2goal(self,condition, easy=False):
        goal = ''
        if condition == 'IsOn_':
            A = random.choice(list(self.GRABBABLE))
            B = random.choice(list(self.SURFACES))
            if B == 'towelrack':
                A = 'towel'
            goal = 'IsOn_' + A + '_' + B
        elif condition == 'IsIn_':
            A = random.choice(list(self.GRABBABLE))
            B = random.choice(list(self.CONTAINERS))
            A = A.split('-')[0]
            B = B.split('-')[0]
            goal += 'IsIn_' + A + '_' + B
            if not easy:
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
        elif condition == 'IsClean_':
            goal = 'IsClean_' + random.choice(list(self.AllObject))
        elif condition == 'IsPlugged_':
            goal = 'IsPlugged_' + random.choice(list(self.HAS_PLUG))
        elif condition == 'IsUnplugged_':
            goal += 'IsUnplugged_' + random.choice(list(self.HAS_PLUG))
        elif condition == 'IsCut_':
            goal = 'IsCut_' + random.choice(list(self.CUTABLE))
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
            goal = self.condition2goal(condition)
            goal_list.append(goal)
        goal_string = ' & '.join(goal_list)
        return goal_string

    def random_generate_goals(self,n, diffcult_type="multi"):
        all_goals = []
        for i in range(n):
            all_goals.append(self.get_goals_string(diffcult_type=diffcult_type))

        return all_goals