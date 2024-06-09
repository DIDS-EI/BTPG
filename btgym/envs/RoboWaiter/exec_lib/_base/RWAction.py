from btgym.behavior_tree.base_nodes import Action
from btgym.behavior_tree import Status
from btgym.behavior_tree.behavior_trees import BehaviorTree

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

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        pass

    def update(self) -> Status:
        # script = [f'<char0> [{self.__class__.__name__.lower()}] <{self.args[0].lower()}> (1)']


        if self.num_args == 1:
            # id = [node['id'] for node in self.env.graph_input['nodes'] if node['class_name'] == self.args[0].lower()][0]
            # script = [f'<char0> [{self.action_class_name.lower()}] <{id}> (1)']
            script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1)']
        else:
            script = [
                f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1) <{self.args[1].lower()}> (1)']

        self.env.run_script(script, verbose=True, camera_mode="PERSON_FROM_BACK")  # FIRST_PERSON
        # print("script: ", script)
        self.change_condition_set()

        return Status.RUNNING
