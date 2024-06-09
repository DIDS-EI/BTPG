from btgym.behavior_tree.base_nodes import Action
from btgym.behavior_tree import Status
from btgym.behavior_tree.behavior_trees import BehaviorTree

class VHTAction(Action):
    can_be_expanded = True
    num_args = 1

    SURFACES = {"kitchentable","towelrack","plate","nightstand","desk","cabinet","bathroomcounter","sofa"}

    # SURFACES = {"kitchentable", "towelrack", "bench", "kitchencabinet", "mousemat", "boardgame", "coffeetable","fryingpan", \
    #             "radio", "cuttingboard", "floor", "tvstand", "bathroomcounter", "oventray", "chair", "kitchencounter","rug", \
    #             "bookshelf", "nightstand", "cabinet", "desk", "stove", "bed", "sofa", "plate", "bathroomcabinet"}
    # 厨房桌子, 毛巾架, 长凳, 厨房橱柜, 鼠标垫, 桌游, 咖啡桌, 煎锅, \
    # 收音机, 切菜板, 地板, 电视架, 浴室台面, 烤箱托盘, 椅子, 厨房台面, 地毯, \
    # 书架, 床头柜, 柜子, 书桌, 炉灶, 床, 沙发, 盘子, 浴室橱柜

    SITTABLE = set()
    # SITTABLE = {"bathtub", "chair", "toilet", "bench", "bed", "rug", "sofa"}
    # 浴缸, 椅子, 厕所, 长凳, 床, 地毯, 沙发

    CAN_OPEN = {"cookingpot", "kitchencabinet", "washingmachine", "window","printer", \
                "curtains", "closet", "box", "microwave", "dishwasher", "radio", "fridge", \
                "garbagecan", "nightstand", "cabinet", "desk", "stove", "door", "folder",
                "clothespile", "bathroomcabinet"}

    # CAN_OPEN = {"coffeemaker", "cookingpot", "toothpaste", "coffeepot", "kitchencabinet", "washingmachine", "window","printer", \
    #             "curtains", "closet", "box", "microwave", "hairproduct", "dishwasher", "radio", "fridge", "toilet","book", \
    #             "garbagecan", "magazine", "nightstand", "cabinet", "milk", "desk", "stove", "door", "folder",
    #             "clothespile", "bathroomcabinet"}
    # 咖啡机, 烹饪锅, 牙膏, 咖啡壶, 厨房橱柜, 洗衣机, 窗户, 打印机, \
    # 窗帘, 衣柜, 盒子, 微波炉, 护发产品, 洗碗机, 收音机, 冰箱, 厕所, 书, \
    # 垃圾桶, 杂志, 床头柜, 柜子, 牛奶, 书桌, 炉灶, 门, 文件夹, 衣物堆, 浴室橱柜

    CONTAINERS = {"washingmachine","dishwasher",
                  "printer","kitchencabinet","garbagecan","clothespile",
                  "fridge","microwave","stove"}
    # CONTAINERS = {"coffeemaker", "kitchencabinet", "washingmachine", "printer", "toaster", "closet", "box", "microwave", \
    #               "dishwasher", "fryingpan", "fridge", "toilet", "garbagecan", "sink", "bookshelf", "nightstand","cabinet", \
    #               "stove", "folder", "clothespile", "bathroomcabinet"}
    # 咖啡机, 厨房橱柜, 洗衣机, 打印机, 烤面包机, 衣柜, 盒子, 微波炉, \
    # 洗碗机, 煎锅, 冰箱, 厕所, 垃圾桶, 水槽, 书架, 床头柜, 柜子, 炉灶, 文件夹, 衣物堆, 浴室橱柜


    GRABBABLE = {
                "bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
                 'cupcake','milk','juice','wine',"cereal",
                 'cutleryknife','fryingpan','dishbowl','plate',
                 'book',"waterglass",'towel',"radio","paper","facecream","clothesshirt","clothespants","facecream",
                  "rag"
             }

    # GRABBABLE = {"sundae", "toothpaste", "clothesshirt", "crackers", "pudding", "alcohol", "boardgame", "wallphone","remotecontrol", \
    #              "whippedcream", "hanger", "cutlets", "candybar", "wine", "toiletpaper", "slippers", "cereal", "apple","magazine", \
    #              "wineglass", "milk", "cupcake", "folder", "wallpictureframe", "cellphone", "coffeepot", "crayons","box", \
    #              "fryingpan", "radio", "chips", "cuttingboard", "lime", "mug", "rug", "carrot", "cutleryfork","clothespile", \
    #              "notes", "plum", "cookingpot", "toy", "salmon", "peach", "condimentbottle", "hairproduct", "salad","mouse", \
    #              "clock", "washingsponge", "bananas", "dishbowl", "oventray", "chocolatesyrup", "creamybuns", "pear","chair", \
    #              "condimentshaker", "bellpepper", "paper", "plate", "facecream", "breadslice", "candle", "towelrack","pancake", \
    #              "cutleryknife", "milkshake", "dishwashingliquid", "keyboard", "towel", "toothbrush", "book", "juice","waterglass", \
    #              "barsoap", "mincedmeat", "clothespants", "chicken", "poundcake", "pillow", "pie",
    #              "rag","duster","papertowel","brush"}
    # 圣代, 牙膏, 衬衫, 饼干, 布丁, 酒精, 桌游, 墙电话, 遥控器, \
    # 鲜奶油, 衣架, 切片肉, 糖果, 酒, 卫生纸, 拖鞋, 麦片, 苹果, 杂志, \
    # 酒杯, 牛奶, 纸杯蛋糕, 文件夹, 墙壁画框, 手机, 咖啡壶, 蜡笔, 盒子, \
    # 煎锅, 收音机, 薯片, 切菜板, 青柠, 杯子, 地毯, 胡落哇, 餐具叉, 衣物堆, \
    # 笔记, 李子, 烹饪锅, 玩具, 鲑鱼, 桃子, 调料瓶, 护发产品, 沙拉, 鼠标, \
    # 时钟, 洗碗海绵, 香蕉, 碗, 烤箱托盘, 巧克力糖浆, 奶油面包, 梨, 椅子, \
    # 调料瓶, 彩椒, 纸张, 盘子, 面霜, 面包片, 蜡烛, 毛巾架, 煎饼, 餐具刀, \
    # 奶昔, 洗碗液, 键盘, 毛巾, 牙刷, 书, 果汁, 水杯, 香皂, 肉末, 裤子, \
    # 鸡肉, 磅蛋糕, 枕头, 馅饼
    # 抹布, 掸子, 纸巾, 刷子

    cleaning_tools = {"rag"}
    # cleaning_tools = {"rag","duster","papertowel","brush"}


    HAS_SWITCH = {"coffeemaker", "cellphone", "candle", "faucet", "washingmachine", "printer", "wallphone","remotecontrol", \
                  "computer", "toaster", "microwave", "dishwasher", "clock", "radio", "lightswitch", "fridge",
                  "tablelamp", "stove", "tv"}
    # 咖啡机, 手机, 蜡烛, 水龙头, 洗衣机, 打印机, 墙电话, 遥控器, \
    # 电脑, 烤面包机, 微波炉, 洗碗机, 时钟, 收音机, 开关, 冰箱, 台灯, 炉灶, 电视

    HAS_PLUG = {"wallphone", "coffeemaker", "lightswitch", "cellphone", "fridge", "toaster", "tablelamp", "microwave", "tv", \
                "clock", "radio", "washingmachine","mouse", "keyboard", "printer"}
    # 墙电话, 咖啡机, 开关, 手机, 冰箱, 烤面包机, 台灯, 微波炉, 电视, \
    # 鼠标, 时钟, 键盘, 收音机, 洗衣机, 打印机

    CUTABLE = set()
    # 无可切割物品

    EATABLE = {"sundae", "breadslice", "whippedcream", "condimentshaker", "chocolatesyrup", "candybar", "creamybuns","pancake", \
               "poundcake", "cereal", "cupcake", "pudding", "salad", "pie", "carrot", "milkshake"}
    # 圣代, 面包片, 鲜奶油, 调料瓶, 巧克力糖浆, 糖果, 奶油面包, 煎饼, \
    # 磅蛋糕, 麦片, 纸杯蛋糕, 布丁, 沙拉, 馅饼, 胡萝卜, 奶昔

    RECIPIENT = {"dishbowl", "wineglass", "coffeemaker", "cookingpot", "box", "mug", "toothbrush", "coffeepot","fryingpan", \
                 "waterglass", "sink", "plate", "washingmachine"}
    # 碗, 酒杯, 咖啡机, 烹饪锅, 盒子, 杯子, 牙刷, 咖啡壶, 煎锅, \
    # 水杯, 水槽, 盘子, 洗衣机

    POURABLE = {"wineglass", "milk", "condimentshaker", "toothpaste", "bottlewater", "mug", "condimentbottle", "hairproduct", \
                "dishwashingliquid", "alcohol", "wine", "juice", "waterglass", "facecream"}
    # 酒杯, 牛奶, 调料瓶, 牙膏, 瓶装水, 杯子, 调料瓶, 护发产品, \
    # 洗碗液, 酒精, 酒, 果汁, 水杯, 面霜

    DRINKABLE = {"milk", "bottlewater", "wine", "alcohol", "juice"}
    # 牛奶, 瓶装水, 酒, 酒精, 果汁

    # switch on #candle  cellphone wallphone washingmachine不行# faucet 浴室龙头
    AllObject = SURFACES | SITTABLE | CAN_OPEN | CONTAINERS | GRABBABLE |\
                 HAS_SWITCH | CUTABLE | EATABLE | RECIPIENT | POURABLE | DRINKABLE

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
