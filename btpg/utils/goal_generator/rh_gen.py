import random
from btpg.utils.goal_generator.goal_gen_base import GoalGenerator
import itertools

class RobotHowGoalGen(GoalGenerator):

    def __init__(self):
        super().__init__()

        self.SURFACES = {"kitchentable", "towelrack", "bench", "kitchencabinet", "mousemat", "boardgame", "coffeetable",
                    "fryingpan", \
                    "radio", "cuttingboard", "floor", "tvstand", "bathroomcounter", "oventray", "chair",
                    "kitchencounter", "rug", \
                    "bookshelf", "nightstand", "cabinet", "desk", "stove", "bed", "sofa", "plate", "bathroomcabinet"}
        # 厨房桌子, 毛巾架, 长凳, 厨房橱柜, 鼠标垫, 桌游, 咖啡桌, 煎锅, \
        # 收音机, 切菜板, 地板, 电视架, 浴室台面, 烤箱托盘, 椅子, 厨房台面, 地毯, \
        # 书架, 床头柜, 柜子, 书桌, 炉灶, 床, 沙发, 盘子, 浴室橱柜

        self.SITTABLE = {"bathtub", "chair", "toilet", "bench", "bed", "rug", "sofa"}
        # 浴缸, 椅子, 厕所, 长凳, 床, 地毯, 沙发

        self.CAN_OPEN = {"coffeemaker", "cookingpot", "toothpaste", "coffeepot", "kitchencabinet", "washingmachine",
                    "window", "printer", \
                    "curtains", "closet", "box", "microwave", "hairproduct", "dishwasher", "radio", "fridge", "toilet",
                    "book", \
                    "garbagecan", "magazine", "nightstand", "cabinet", "milk", "desk", "stove", "door", "folder",
                    "clothespile", "bathroomcabinet", "oven"}
        # 咖啡机, 烹饪锅, 牙膏, 咖啡壶, 厨房橱柜, 洗衣机, 窗户, 打印机, \
        # 窗帘, 衣柜, 盒子, 微波炉, 护发产品, 洗碗机, 收音机, 冰箱, 厕所, 书, \
        # 垃圾桶, 杂志, 床头柜, 柜子, 牛奶, 书桌, 炉灶, 门, 文件夹, 衣物堆, 浴室橱柜, 烤箱

        self.CONTAINERS = {"coffeemaker", "kitchencabinet", "washingmachine", "printer", "toaster", "closet", "box",
                      "microwave", \
                      "dishwasher", "fryingpan", "fridge", "toilet", "garbagecan", "sink", "bookshelf", "nightstand",
                      "cabinet", \
                      "stove", "folder", "clothespile", "bathroomcabinet", "oven", "cookingpot", "desk"}
        # 咖啡机, 厨房橱柜, 洗衣机, 打印机, 烤面包机, 衣柜, 盒子, 微波炉, \
        # 洗碗机, 煎锅, 冰箱, 厕所, 垃圾桶, 水槽, 书架, 床头柜, 柜子, 炉灶, 文件夹, 衣物堆, 浴室橱柜

        self.GRABBABLE = {"sundae", "toothpaste", "clothesshirt", "crackers", "pudding", "alcohol", "boardgame", "wallphone",
                     "remotecontrol", \
                     "whippedcream", "hanger", "cutlets", "candybar", "wine", "toiletpaper", "slippers", "cereal",
                     "apple", "magazine", \
                     "wineglass", "milk", "cupcake", "folder", "wallpictureframe", "cellphone", "coffeepot", "crayons",
                     "box", \
                     "fryingpan", "radio", "chips", "cuttingboard", "lime", "mug", "rug", "carrot", "cutleryfork",
                     "clothespile", \
                     "notes", "plum", "cookingpot", "toy", "salmon", "peach", "condimentbottle", "hairproduct", "salad",
                     "mouse", \
                     "clock", "washingsponge", "bananas", "dishbowl", "oventray", "chocolatesyrup", "creamybuns",
                     "pear", "chair", \
                     "condimentshaker", "bellpepper", "paper", "plate", "facecream", "breadslice", "candle",
                     "towelrack", "pancake", \
                     "cutleryknife", "kitchenknife", "milkshake", "dishwashingliquid", "keyboard", "towel",
                     "toothbrush", "book", "juice", "waterglass", \
                     "barsoap", "mincedmeat", "clothespants", "chicken", "poundcake", "pillow", "pie",
                     "rag", "duster", "papertowel", "brush"}
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

        # cleaning_tools = {"rag", "duster", "papertowel", "brush"}
        # cutting_tools={"cutleryknife","kitchenknife"}
        cleaning_tools = {"rag"}
        cutting_tools = {"kitchenknife"}

        self.HAS_SWITCH = {"coffeemaker", "cellphone", "candle", "faucet", "washingmachine", "printer", "wallphone",
                      "remotecontrol", \
                      "computer", "toaster", "microwave", "dishwasher", "clock", "radio", "lightswitch", "fridge",
                      "tablelamp", "stove", "tv", "oven"}
        # 咖啡机, 手机, 蜡烛, 水龙头, 洗衣机, 打印机, 墙电话, 遥控器, \
        # 电脑, 烤面包机, 微波炉, 洗碗机, 时钟, 收音机, 开关, 冰箱, 台灯, 炉灶, 电视

        self.HAS_PLUG = {"wallphone", "coffeemaker", "lightswitch", "cellphone", "fridge", "toaster", "tablelamp",
                    "microwave", "tv", \
                    "clock", "radio", "washingmachine", "mouse", "keyboard", "printer", "oven", "dishwasher"}
        # 墙电话, 咖啡机, 开关, 手机, 冰箱, 烤面包机, 台灯, 微波炉, 电视, \
        # 鼠标, 时钟, 键盘, 收音机, 洗衣机, 打印机

        self.EATABLE = {"sundae", "breadslice", "whippedcream", "condimentshaker", "chocolatesyrup", "candybar",
                   "creamybuns", "pancake", \
                   "poundcake", "cereal", "cupcake", "pudding", "salad", "pie", "carrot", "milkshake"}
        # 圣代, 面包片, 鲜奶油, 调料瓶, 巧克力糖浆, 糖果, 奶油面包, 煎饼, \
        # 磅蛋糕, 麦片, 纸杯蛋糕, 布丁, 沙拉, 馅饼, 胡萝卜, 奶昔

        # CUTABLE = set()
        self.CUTABLE = {"apple", "bananas", "breadslice", "cutlets", "poundcake", "pancake", "pie", "carrot", "chicken",
                   "lime", "salmon", "peach", \
                   "pear", "plum", "bellpepper"}
        # 无可切割物品

        self.WASHABLE = {"apple", "bananas", "carrot", "chicken", "lime", "salmon", "peach", "pear", "plum", "rag",
                    "cutlets"}

        self.RECIPIENT = {"dishbowl", "wineglass", "coffeemaker", "cookingpot", "box", "mug", "toothbrush", "coffeepot",
                     "fryingpan", \
                     "waterglass", "sink", "plate", "washingmachine"}
        # 碗, 酒杯, 咖啡机, 烹饪锅, 盒子, 杯子, 牙刷, 咖啡壶, 煎锅, \
        # 水杯, 水槽, 盘子, 洗衣机

        self.POURABLE = {"wineglass", "milk", "condimentshaker", "toothpaste", "bottlewater", "mug", "condimentbottle",
                    "hairproduct", \
                    "dishwashingliquid", "alcohol", "wine", "juice", "waterglass", "facecream"}
        # 酒杯, 牛奶, 调料瓶, 牙膏, 瓶装水, 杯子, 调料瓶, 护发产品, \
        # 洗碗液, 酒精, 酒, 果汁, 水杯, 面霜

        self.DRINKABLE = {"milk", "bottlewater", "wine", "alcohol", "juice"}
        # 牛奶, 瓶装水, 酒, 酒精, 果汁

        # switch on #candle  cellphone wallphone washingmachine不行# faucet 浴室龙头
        self.AllObject = self.SURFACES | self.SITTABLE | self.CAN_OPEN | self.CONTAINERS | self.GRABBABLE | \
                    self.HAS_SWITCH | self.CUTABLE | self.EATABLE | self.RECIPIENT | self.POURABLE | self.DRINKABLE

        self.cond_pred = {'IsOn_', 'IsIn_', 'IsOpen_', 'IsSwitchedOn_', 'IsClean_',
                     'IsPlugged_', 'IsCut_', 'IsNear_self_'}


        self.set_1_food = self.GRABBABLE & (self.EATABLE|self.DRINKABLE|{"apple","bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
                     'milk','wine',"cereal","plate","lime","salmon", "peach","pear","plum"})
        self.set_2_cloth =  self.GRABBABLE & {"clothespile","clothesshirt","clothespants"}
        self.putIn_valid_args = set(list(itertools.product(self.set_1_food, \
                                            self.CONTAINERS - {"washingmachine","dishwasher","printer","folder","closet","clothespile"})) \
                     + list(itertools.product(self.GRABBABLE-self.set_1_food-self.set_2_cloth,\
                        self.CONTAINERS-{"fridge","microwave","stove","fryingpan","closet","clothespile","washingmachine","dishwasher","printer","folder"})) \
                    + list(itertools.product(self.set_2_cloth,self.CONTAINERS-{"fridge","microwave","stove","fryingpan","dishwasher","printer","folder"})) \
                + list(itertools.product(self.GRABBABLE & {"dishbowl","plate"}, {"dishwasher"})) \
                + list(itertools.product(self.GRABBABLE & {"paper"}, {"printer","folder"})) \
                + list(itertools.product(self.GRABBABLE & {"papertowel"}, {"garbagecan"})))



        self.put_valid_args = set(list(itertools.product(self.GRABBABLE-self.set_1_food, self.SURFACES-{"towelrack","plate","fryingpan"})) \
                        + list(itertools.product(self.GRABBABLE & {'towel'}, {"towelrack"})) \
                        + list(itertools.product(self.set_1_food, self.SURFACES-{"towelrack","bathroomcounter"})))



    def condition2goal(self,condition):
        goal = ''
        if condition == 'IsOn_':
            # A = random.choice(list(self.GRABBABLE))
            # B = random.choice(list(self.SURFACES))
            random_choice = random.choice(list(self.put_valid_args))
            A, B = random_choice
            goal = 'IsOn_' + A + '_' + B
        elif condition == 'IsIn_':
            # A = random.choice(list(self.GRABBABLE))
            # A = A.split('-')[0]
            # # B = random.choice(list(self.CONTAINERS))
            # # B = B.split('-')[0]
            random_choice = random.choice(list(self.putIn_valid_args))
            A, B = random_choice
            goal += 'IsIn_' + A + '_' + B
            if B in self.CAN_OPEN:
                goal += ' & IsClose_' + B
        elif condition == 'IsOpen_':
            goal = 'IsOpen_' + random.choice(list(self.CAN_OPEN))
        elif condition == 'IsClose_':
            # First, choose one of the two lists
            chosen_list = random.choice([list(self.WASHABLE), list(self.AllObject - self.WASHABLE)])
            # Then, select an item from the chosen list
            selected_item = random.choice(chosen_list)
            goal = 'IsClose_' +selected_item
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