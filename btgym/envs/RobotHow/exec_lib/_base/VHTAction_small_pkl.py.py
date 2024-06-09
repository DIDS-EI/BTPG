from btgym.behavior_tree.base_nodes import Action
from btgym.behavior_tree import Status
from btgym.behavior_tree.behavior_trees import BehaviorTree
from btgym.utils import ROOT_PATH
class VHTAction_small(Action):
    can_be_expanded = True
    num_args = 1


    import pickle
    ctg_objs_path = f"{ROOT_PATH}/../test/BT_EXP/ctg_objs.pickle"
    # 打开之前写入的文件，注意使用二进制模式读取
    with open(ctg_objs_path, 'rb') as file:
        # 使用pickle.load()函数从文件加载数据
        categories_objs_dic = pickle.load(file)

    SURFACES = categories_objs_dic['SURFACES']
    SITTABLE = categories_objs_dic['SITTABLE']
    CAN_OPEN = categories_objs_dic['CAN_OPEN']
    CONTAINERS = categories_objs_dic['CONTAINERS']
    GRABBABLE = categories_objs_dic['GRABBABLE']
    cleaning_tools = categories_objs_dic['cleaning_tools']
    cutting_tools = categories_objs_dic['cutting_tools']
    HAS_SWITCH = categories_objs_dic['HAS_SWITCH']
    HAS_PLUG = categories_objs_dic['HAS_PLUG']
    CUTABLE = categories_objs_dic['CUTABLE']
    EATABLE = categories_objs_dic['EATABLE']
    WASHABLE = categories_objs_dic['WASHABLE']  # Corrected typo from WASHBLE to WASHABLE
    RECIPIENT = categories_objs_dic['RECIPIENT']
    POURABLE = categories_objs_dic['POURABLE']
    DRINKABLE = categories_objs_dic['DRINKABLE']

    AllObject = SURFACES | SITTABLE | CAN_OPEN | CONTAINERS | GRABBABLE |\
                 HAS_SWITCH | CUTABLE | EATABLE | RECIPIENT | POURABLE | DRINKABLE


    # categories = ['SURFACES','SITTABLE', 'CAN_OPEN', 'CONTAINERS', 'GRABBABLE', 'cleaning_tools', \
    #      'cutting_tools', 'HAS_SWITCH', 'HAS_PLUG', 'CUTABLE', 'EATABLE', 'WASHBLE', 'RECIPIENT', \
    #      'POURABLE', 'DRINKABLE']
    # print(globals())
    # 使用字典推导式创建变量字典
    # ctg_name_dic = {category: globals()[category] for category in categories}
    # 创建一个字典，存储每个类别对应的集合对象
    # ctg_ls = [SURFACES,SITTABLE,CAN_OPEN,CONTAINERS,GRABBABLE,cleaning_tools,\
    #     cutting_tools,HAS_SWITCH,HAS_PLUG,CUTABLE,EATABLE,WASHBLE,RECIPIENT,POURABLE,\
    #     DRINKABLE]
    # ctg_name_ls = ['SURFACES','SITTABLE', 'CAN_OPEN', 'CONTAINERS', 'GRABBABLE', 'cleaning_tools', \
    #          'cutting_tools', 'HAS_SWITCH', 'HAS_PLUG', 'CUTABLE', 'EATABLE', 'WASHABLE', 'RECIPIENT', \
    #          'POURABLE', 'DRINKABLE']

    # import pickle
    # ctg_objs_path = f"{ROOT_PATH}/../test/EXP/ctg_objs.pickle"
    # # 打开之前写入的文件，注意使用二进制模式读取
    # with open(ctg_objs_path, 'rb') as file:
    #     # 使用pickle.load()函数从文件加载数据
    #     categories_objs_dic = pickle.load(file)
    #
    # for ctg,name in zip(ctg_ls,ctg_name_ls):
    #     ctg = categories_objs_dic[name]
    #
    # AllObject = SURFACES | SITTABLE | CAN_OPEN | CONTAINERS | GRABBABLE |\
    #              HAS_SWITCH | CUTABLE | EATABLE | RECIPIENT | POURABLE | DRINKABLE



    # def __init__(self):
    #     super().__init__()  # 调用父类的初始化方法
    #     ctg_objs_path = f"{ROOT_PATH}/../test/EXP/ctg_objs.pickle"
    #     self.reload_objs_ls(objs_path)  # 在初始化方法中调用 reload_objs_ls 方法
    #
    # def reload_objs_ls(self,objs_path):
    #     categories = ['SURFACES','SITTABLE', 'CAN_OPEN', 'CONTAINERS', 'GRABBABLE', 'cleaning_tools', \
    #          'cutting_tools', 'HAS_SWITCH', 'HAS_PLUG', 'CUTABLE', 'EATABLE', 'WASHBLE', 'RECIPIENT', \
    #          'POURABLE', 'DRINKABLE']
    #     import pickle
    #     # 打开之前写入的文件，注意使用二进制模式读取
    #     with open(behavior_lib_path, 'rb') as file:
    #         # 使用pickle.load()函数从文件加载数据
    #         categories_objs_dic = pickle.load(file)
    #
    #     for ctg in categories:
    #         # self.ctg = categories_objs_dic[ctg]
    #         setattr(self, ctg, categories_objs_dic[ctg])
    #
    #     self.AllObject = self.SURFACES | self.SITTABLE | self.CAN_OPEN | self.CONTAINERS | self.GRABBABLE | \
    #                 self.HAS_SWITCH | self.CUTABLE | self.EATABLE | self.RECIPIENT | self.POURABLE | self.DRINKABLE

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
