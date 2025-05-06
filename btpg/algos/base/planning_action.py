
import copy
import random


#定义行动类，行动包括前提、增加和删除影响
# Define action categories, which include prerequisites, adding and deleting impacts
class PlanningAction:
    def __init__(self,name='anonymous action',pre=set(),add=set(),del_set=set(),cost=10,vaild_num=0,vild_args=set()):
        self.pre=copy.deepcopy(pre)
        self.add=copy.deepcopy(add)
        self.del_set=copy.deepcopy(del_set)
        self.name=name
        self.real_cost=cost
        self.cost=cost
        self.priority = cost
        self.vaild_num=vaild_num
        self.vild_args = vild_args

    def __str__(self):
        return self.name

    def generate_from_state_local(self,state,literals_num_set,all_obj_set=set(),obj_num=0,obj=None):
        pre_num = random.randint(0, len(state))
        self.pre = set(random.sample(state, pre_num))

        add_set = literals_num_set - self.pre
        add_num = random.randint(0, len(add_set))
        self.add = set(random.sample(add_set, add_num))

        del_set = literals_num_set - self.add
        del_num = random.randint(0, len(del_set))
        self.del_set = set(random.sample(del_set, del_num))

        if all_obj_set!=set():
            self.vaild_num = random.randint(1, obj_num-1)
            self.vild_args = (set(random.sample(all_obj_set, self.vaild_num)))
            if obj!=None:
                self.vild_args.add(obj)
                self.vaild_num = len(self.vild_args)

    def update(self,name,pre,del_set,add):
        self.name = name
        self.pre = pre
        self.del_set = del_set
        self.add = add
        return self


    def print_action(self):
        print (self.pre)
        print(self.add)
        print(self.del_set)

    # 从状态随机生成一个行动
    # def generate_from_state(self,state,num):
    #     for i in range(0,num):
    #         if i in state:
    #             if random.random() >0.5:
    #                 self.pre.add(i)
    #                 if random.random() >0.5:
    #                     self.del_set.add(i)
    #                 continue
    #         if random.random() > 0.5:
    #             self.add.add(i)
    #             continue
    #         if random.random() >0.5:
    #             self.del_set.add(i)

    # def generate_from_state_local(self,literals_num_set):
    #     # pre_num = random.randint(0, min(pre_max, len(state)))
    #     # self.pre = set(np.random.choice(list(state), pre_num, replace=False))
    #     #
    #     # add_set = literals_num_set - self.pre
    #     # add_num = random.randint(0, len(add_set))
    #     # self.add = set(np.random.choice(list(add_set), add_num, replace=False))
    #     #
    #     # del_set = literals_num_set - self.add
    #     # del_num = random.randint(0, len(del_set))
    #     # self.del_set = set(np.random.choice(list(del_set), del_num, replace=False))
    #
    #     pre_num = random.randint(0, len(state))
    #     self.pre = set(random.sample(state, pre_num))
    #
    #     add_set = literals_num_set - self.pre
    #     add_num = random.randint(0, len(add_set))
    #     self.add = set(random.sample(add_set, add_num))
    #
    #     del_set = literals_num_set - self.add
    #     del_num = random.randint(0, len(del_set))
    #     self.del_set = set(random.sample(del_set, del_num))





#生成随机状态
def generate_random_state(num):
    result = set()
    for i in range(0,num):
        if random.random()>0.5:
            result.add(i)
    return result

# #从状态和行动生成后继状态
def state_transition(state,action):
    if not action.pre <= state:
        print ('error: action not applicable')
        return state
    new_state=(state | action.add) - action.del_set
    return new_state