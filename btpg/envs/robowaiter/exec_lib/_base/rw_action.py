from btpg.behavior_tree.base_nodes import Action
from btpg.behavior_tree import Status
from btpg.behavior_tree.behavior_trees import BehaviorTree

class RWAction(Action):
    can_be_expanded = True
    num_args = 1

    tables_for_placement = {'Bar', 'Bar2', 'WaterStation', 'CoffeeStation', 'Table1', 'Table2', 'Table3','BrightTable6'}
    all_object = {
        'Coffee', 'Water', 'Dessert', 'Softdrink', 'BottledDrink', 'Yogurt', 'ADMilk', 'MilkDrink', 'Milk','VacuumCup',
        'Chips', 'NFCJuice', 'Bernachon', 'SpringWater'}
    tables_for_guiding = {"QuietTable1","QuietTable2",
                          "BrightTable1","BrightTable2","BrightTable3","BrightTable4","BrightTable5","BrightTable6",
                          'CoffeeTable','WaterTable','Table1', 'Table2', 'Table3'}

    all_place = tables_for_guiding | tables_for_placement

    SURFACES = tables_for_placement
    GRABBABLE =all_object
    AllObject = tables_for_placement | all_object | tables_for_guiding



    num_of_obj_on_place={
        'Bar': 0,  # (247.0, 520.0, 100.0)
        'Bar2': 0,
        'WaterStation': 0,
        'CoffeeStation': 0,
        'Table1': 0,
        'Table2': 0,
        'Table3': 0,
        'BrightTable6': 0,
    }

    place_xyz_dic={
        'Bar': (247.0, 520.0, 100.0), #(247.0, 520.0, 100.0)
        'Bar2': (240.0, 40.0, 100.0),
        'WaterStation':(-70.0, 500.0, 107),
        'CoffeeStation':(250.0, 310.0, 100.0),
        'Table1': (340.0, 900.0, 99.0),
        'Table2': (-55.0, 0.0, 107),
        'Table3':(-55.0, 150.0, 107),
        'BrightTable6': (5, -315, 116.5),
    }

    place_have_obj_xyz_dic = {
        'QuietTable1': (480, 1300, 70),
        'QuietTable2': (250, -240, 70),
        'BrightTable1': (230, 1200, 35),
        'BrightTable2': (65, 1000, 35),
        'BrightTable3': (-80, 850, 35),
        'BrightTable4': (-270, 520, 70),
        'BrightTable5': (-270, 420, 35)
    }
    place_have_obj_xyz_dic.update(place_xyz_dic)

    place_en2zh_name={
        'Bar': "吧台",
        'Bar2': "另一侧的吧台",
        'WaterStation': "大厅的茶水桌",
        'CoffeeStation': "咖啡桌",
        'Table1': "前门斜桌子",
        'Table2': "大厅长桌子西侧",
        'Table3': "大厅长桌子东侧",
        'BrightTable6': "后门靠窗边圆桌",
        'QuietTable1': "前门角落双人圆桌",
        'QuietTable2': "后门角落三人圆桌",
        'BrightTable1': "靠窗边第一个四人矮桌",
        'BrightTable2': "靠窗边第二个四人矮桌",
        'BrightTable3': "靠窗边第三个四人矮桌",
        'BrightTable4': "大厅里靠窗边长桌子",
        'BrightTable5': "大厅里靠窗边多人矮桌",
    }

    place_xy_yaw_dic={
        'Bar': (247.0, 520.0, 180),  # (247.0, 520.0, 100.0)
        'Bar2': (240.0, 40.0, 100.0),
        'WaterStation': (-70.0, 500.0, 107),
        'CoffeeStation': (250.0, 310.0, 100.0),
        'Table1': (340.0, 900.0, 99.0),
        'Table2': (-55.0, 0.0, 107),
        'Table3': (-55.0, 150.0, 107),
        'BrightTable6': (5, -315, 116.5),

        'QuietTable1':(480,1300,90),
        'QuietTable2':(250,-240,-65),
        'BrightTable1':(230,1200,-135),
        'BrightTable2': (65, 1000, 135),
        'BrightTable3': (-80, 850, 135),
        'BrightTable4': (-270, 520, 150),
        'BrightTable5': (-270, 420, 90) #(-270, 420, -135)
    }
    container_dic={
        'Coffee':'CoffeeCup',
        'Water': 'Glass',
        'Dessert':'Plate'
    }


    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        pass

    def update(self) -> Status:

        self._update()
        self.change_condition_set()

        return Status.RUNNING
