from btpg.behavior_tree.base_nodes import Action
from btpg.behavior_tree import Status
from btpg.behavior_tree.behavior_trees import BehaviorTree

class OGAction(Action):
    can_be_expanded = True
    num_args = 1

    # SURFACES = {"kitchentable"}
    #
    # SITTABLE =  set()
    #
    # CAN_OPEN = {"fridge","window"}
    # CONTAINERS = {"fridge","garbagecan"}
    #
    #
    # GRABBABLE = {"apple",'breadslice','milk','plate',"rag","kitchenknife"}
    #
    # cleaning_tools = {"rag"}
    # cutting_tools = {"kitchenknife"}
    #
    # HAS_SWITCH = {"tv","faucet","candle"}
    #
    # HAS_PLUG = {"tv","mouse","fridge"}
    # # 墙电话, 咖啡机, 开关, 手机, 冰箱, 烤面包机, 台灯, 微波炉, 电视, \
    # # 鼠标, 时钟, 键盘, 收音机, 洗衣机, 打印机
    #
    # CUTABLE = {"apple",'breadslice'}
    #
    # WASHABLE={"apple"}
    #
    # EATABLE = {"apple",'breadslice'}
    #
    # DRINKABLE = {'milk'}


    # SURFACES = {"kitchentable", "desk", "coffeetable", "bed"}
    # SITTABLE = {"bed"}
    # CAN_OPEN = {"fridge", "window","washingmachine"}
    # CONTAINERS = {"fridge", "garbagecan","washingmachine"}
    # GRABBABLE = {"apple", 'breadslice', 'wine', 'plate', "rag", "kitchenknife", "pear", "cutlets"}
    # cleaning_tools = {"rag"}
    # cutting_tools = {"kitchenknife"}
    # HAS_SWITCH = {"tv", "faucet", "candle","washingmachine"}
    # HAS_PLUG = {"tv", "mouse", "fridge","washingmachine"}
    # # 墙电话, 咖啡机, 开关, 手机, 冰箱, 烤面包机, 台灯, 微波炉, 电视, \
    # # 鼠标, 时钟, 键盘, 收音机, 洗衣机, 打印机
    # CUTABLE = {"apple", 'breadslice', "pear", "cutlets"}
    # WASHABLE = {"apple", "rag", "kitchenknife", "pear", "cutlets"}
    # EATABLE = {"apple", 'breadslice'}
    # DRINKABLE = {'wine'}




    # SURFACES = {"kitchencabinet", "bed", "kitchentable"}  
    #             # bottom_cabinet_slgzfc_0， bed_zrumze_0， breakfast_table_skczfi_1   #coffee_table_fqluyq_0
    # SITTABLE = {"bed"}          
    # CAN_OPEN = {"fridge", "window", "microwave", "kitchencabinet"} #fridge_xyejdx_0, window_ithrgo_2, microwave_182, 
    # CONTAINERS = {"fridge", "garbagecan", "microwave", "kitchencabinet"} # 
    # GRABBABLE = {"apple", 'wine', 'plate', "rag", "kitchenknife", "cutlets"}
    #             # apple_omzprq_0
    # cleaning_tools = {"rag"}
    # cutting_tools = {"kitchenknife"}
    # HAS_SWITCH = {"tv", "faucet", "candle", "microwave"} # electric_switch_wseglt_2
    # HAS_PLUG = {"tv", "mouse", "fridge", "microwave"}
    # CUTABLE = {"apple", "cutlets"} # apple,
    # WASHABLE = {"apple", "rag", "kitchenknife", "cutlets"}
    # EATABLE = {"apple", 'cutlets'}
    # DRINKABLE = {'wine'}


    SURFACES = {"bottomcabinet", "bed", "breakfasttable"}  
                # bottom_cabinet_slgzfc_0， bed_zrumze_0， breakfast_table_skczfi_1   #coffee_table_fqluyq_0
    SITTABLE = {"bed"}          
    CAN_OPEN = {"fridge", "window", "microwave", "bottomcabinet"} #fridge_xyejdx_0, window_ithrgo_2, microwave_hjjxmi_0, 
    CONTAINERS = {"fridge", "garbagecan", "microwave", "bottomcabinet"} # trash_can_zotrbg_0
    GRABBABLE = {"apple", 'milk', 'plate', "rag", "kitchenknife", "chips"} 
                # apple_omzprq_0, box_of_almond_milk_oiiqwq_0, plate_amhlqh_0,carving_knife_alekva_0, bag_of_chips_bryahw_0
    cleaning_tools = {"rag"} # rag_oocmed_0
    cutting_tools = {"kitchenknife"} # carving_knife_alekva_0
    HAS_SWITCH = {"tv", "faucet", "electricswitch", "microwave"} # standing_tv_udotid_0, sink_zexzrc_0, electric_switch_wseglt_2
    HAS_PLUG = {"tv", "loudspeaker", "fridge", "microwave"} #loudspeaker_bmpdyv_0
    CUTABLE = {"apple", "chips"} # bag_of_chips_bryahw_0
    WASHABLE = {"apple", "rag", "kitchenknife", "chips"}
    EATABLE = {"apple", 'chips'}
    DRINKABLE = {'milk'} #box_of_almond_milk_oiiqwq_0

    obj2og = {}
    obj2og["apple"] = "apple_omzprq_0"
    obj2og["milk"] = "box_of_almond_milk_oiiqwq_0"
    obj2og["chips"] = "bag_of_chips_bryahw_0"

    obj2og["microwave"] = "microwave_hjjxmi_0"
    obj2og["fridge"] = "fridge_xyejdx_0"
    obj2og["window"] = "window_ithrgo_2"
    obj2og["bottomcabinet"] = "bottom_cabinet_slgzfc_0"
    obj2og["bed"] = "bed_zrumze_0"
    obj2og["breakfasttable"] = "breakfast_table_skczfi_1"
    obj2og["rag"] = "rag_oocmed_0"
    obj2og["kitchenknife"] = "carving_knife_alekva_0"
    obj2og["plate"] = "plate_amhlqh_0"
    obj2og["tv"] = "standing_tv_udotid_0"
    obj2og["faucet"] = "sink_zexzrc_0"
    obj2og["electricswitch"] = "electric_switch_wseglt_2"
    obj2og["loudspeaker"] = "loudspeaker_bmpdyv_0"
    obj2og["garbagecan"] = "trash_can_zotrbg_0"
    





    AllObject = SURFACES | SITTABLE | CAN_OPEN | CONTAINERS | GRABBABLE |\
                 HAS_SWITCH | HAS_PLUG| CUTABLE | WASHABLE|cleaning_tools|cutting_tools

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
