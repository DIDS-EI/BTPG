from btgym.behavior_tree.base_nodes import Action
from btgym.behavior_tree import Status
from btgym.behavior_tree.behavior_trees import BehaviorTree

class VHTAction_small(Action):
    can_be_expanded = True
    num_args = 1

    SURFACES = {"kitchentable"}

    SITTABLE = { "bed"}

    CAN_OPEN = {"fridge","window"}
    CONTAINERS = {"fridge","garbagecan"}


    GRABBABLE = {"apple",'breadslice','milk','plate',"rag","kitchenknife"}

    cleaning_tools = {"rag"}
    cutting_tools = {"kitchenknife"}

    HAS_SWITCH = {"tv","faucet","candle"}

    HAS_PLUG = {"tv","mouse","fridge"}
    # 墙电话, 咖啡机, 开关, 手机, 冰箱, 烤面包机, 台灯, 微波炉, 电视, \
    # 鼠标, 时钟, 键盘, 收音机, 洗衣机, 打印机

    CUTABLE = {"apple",'breadslice'}

    WASHABLE={"apple"}

    EATABLE = {"apple",'breadslice'}

    DRINKABLE = {'milk'}


    AllObject = SURFACES | SITTABLE | CAN_OPEN | CONTAINERS | GRABBABLE |\
                 HAS_SWITCH | CUTABLE

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
