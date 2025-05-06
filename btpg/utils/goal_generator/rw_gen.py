import random
from btpg.utils.goal_generator.goal_gen_base import GoalGenerator


class RoboWaiterGoalGen(GoalGenerator):

    def __init__(self):
        super().__init__()
        self.tables_for_placement = {'Bar', 'Bar2', 'WaterStation', 'CoffeeStation', 'Table1', 'Table2', 'Table3',
                                'BrightTable6'}
        self.all_object = {
            'Coffee', 'Water', 'Dessert', 'Softdrink', 'BottledDrink', 'Yogurt', 'ADMilk', 'MilkDrink', 'Milk',
            'VacuumCup',
            'Chips', 'NFCJuice', 'Bernachon', 'SpringWater'}
        self.tables_for_guiding = {"QuietTable1", "QuietTable2",
                              "BrightTable1", "BrightTable2", "BrightTable3", "BrightTable4", "BrightTable5",
                              "BrightTable6",
                              'CoffeeTable', 'WaterTable', 'Table1', 'Table2', 'Table3'}

        self.all_place = self.tables_for_guiding | self.tables_for_placement

        self.cond_pred={"RobotNear_", "On_", "Holding_", "Exists_", "IsClean_", "Active_", "Closed_", "Low_"}

    def condition2goal(self,condition):
        goal = ''
        if condition == 'On_':
            A = random.choice(list(self.all_object))
            B = random.choice(list(self.tables_for_placement))
            goal = 'On_' + A + '_' + B
        elif condition == 'RobotNear_':
            goal = 'RobotNear_' + random.choice(list(self.all_place))
        elif condition == 'Holding_':
            goal = 'Holding_' + random.choice(list(self.all_object))
        elif condition == 'Exists_':
            goal = 'Exists_' + random.choice(["Coffee","Water","Dessert"])
        elif condition == 'IsClean_':
            goal = 'IsClean_' + random.choice(['Table1','Floor','Chairs'])
        elif condition == 'Active_':
            goal = 'Active_' + random.choice(['AC','TubeLight','HallLight'])
        elif condition == 'Closed_':
            goal = 'Closed_Curtain'
        elif condition == 'Low_':
            goal = 'Low_ACTemperature'
        return goal


