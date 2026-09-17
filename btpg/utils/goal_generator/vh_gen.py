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

    @staticmethod
    def goal_slot(goal):
        """The mutex slot a conjunct belongs to; two conjuncts of the same slot
        cannot hold together, so a goal must not contain more than one of them.

        The slots follow the action model: `Walk` deletes every other
        `IsNear(self,*)`, the grab and put actions delete every other location of
        the object they move, and `Open`/`Close` and `SwitchOn`/`SwitchOff` delete
        each other.
        """
        if goal.startswith('IsNear_self_'):
            return 'agent'
        for p in ('IsOn_', 'IsIn_'):
            if goal.startswith(p):
                return 'location:' + goal[len(p):].rsplit('_', 1)[0]
        for p in ('IsOpen_', 'IsClose_'):
            if goal.startswith(p):
                return 'door:' + goal[len(p):]
        for p in ('IsSwitchedOn_', 'IsSwitchedOff_'):
            if goal.startswith(p):
                return 'switch:' + goal[len(p):]
        return goal

    def condition2goal(self,condition,diffcult_type="multi"):
        """One conjunct for `condition`, or two joined by ' & ' for the
        put-in-and-close pair.

        Every choice is made over a sorted sequence: `random.choice(list(a_set))`
        depends on the iteration order of the set, which varies with
        PYTHONHASHSEED, so seeding `random` would not make generation
        reproducible across processes.
        """
        goal = ''
        if condition == 'IsOn_':
            A = random.choice(sorted(self.GRABBABLE))
            # `plate` is both grabbable and a surface; an object cannot rest on itself.
            surfaces = sorted(s for s in self.SURFACES if s != A)
            if not surfaces:
                return ''
            B = random.choice(surfaces)
            goal = 'IsOn_' + A + '_' + B
        elif condition == 'IsIn_':
            A = random.choice(sorted(self.GRABBABLE))
            containers = sorted(c for c in self.CONTAINERS if c != A)
            if not containers:
                return ''
            B = random.choice(containers)
            A = A.split('-')[0]
            B = B.split('-')[0]
            goal += 'IsIn_' + A + '_' + B
            if diffcult_type!="single":
                if B in self.CAN_OPEN:
                    goal += ' & IsClose_' + B
        elif condition == 'IsOpen_':
            goal = 'IsOpen_' + random.choice(sorted(self.CAN_OPEN))
        elif condition == 'IsClose_':
            goal = 'IsClose_' + random.choice(sorted(self.CAN_OPEN))
        elif condition == 'IsSwitchedOn_':
            A = random.choice(sorted(self.HAS_SWITCH))
            goal += 'IsSwitchedOn_' + A
        elif condition == 'IsSwitchedOff_':
            goal += 'IsSwitchedOff_' + random.choice(sorted(self.HAS_SWITCH))
        elif condition == 'IsNear_self_':
            goal = 'IsNear_self_' + random.choice(sorted(self.AllObject))
        return goal


    def get_goals_string(self,diffcult_type="multi",max_conjuncts=3,max_tries=8):
        """A conjunctive goal whose conjuncts occupy distinct mutex slots.

        Conjuncts are drawn as before, but a draw is kept only if its slot is
        still free, so the result cannot ask for the agent to stand in two
        places, an object to be in two places, or a container to be open and
        closed at once. The put-in-and-close pair is kept together: it is added
        only if both of its slots are free and both fit within the budget.
        """
        if diffcult_type == "single":
            goal_mount = random.randint(1, 1)
        elif diffcult_type == "multi":
            goal_mount = random.randint(2, 3)
        elif diffcult_type == "mix":
            goal_mount = random.randint(1, 3)

        goal_list, used = [], set()
        for _ in range(goal_mount):
            if len(goal_list) >= max_conjuncts:
                break
            for _try in range(max_tries):
                condition = random.choice(sorted(self.cond_pred))
                goal = self.condition2goal(condition,diffcult_type=diffcult_type)
                if not goal:
                    continue
                parts = goal.split(' & ')
                slots = [self.goal_slot(p) for p in parts]
                if len(set(slots)) != len(slots) or used & set(slots):
                    continue
                if len(goal_list) + len(parts) > max_conjuncts:
                    # Never split the pair; drop the optional half and retry the
                    # main conjunct on its own.
                    parts, slots = parts[:1], slots[:1]
                    if len(goal_list) + 1 > max_conjuncts or slots[0] in used:
                        continue
                goal_list.extend(parts)
                used.update(slots)
                break

        goal_string = ' & '.join(goal_list)
        return goal_string
