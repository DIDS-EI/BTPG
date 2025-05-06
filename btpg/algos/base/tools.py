
import re
# from tabulate import tabulate
import numpy as np
import random
import time

# from btpg.algos.bt_planning.Action import Action,generate_random_state,state_transition
# from btpg.algos.bt_planning.OptimalBTExpansionAlgorithm import OptBTExpAlgorithm

from btpg.algos.base.behaviour_tree import Leaf, ControlBT


def set_to_tuple(s):
    """
    Convert a set of strings to a tuple with elements sorted.
    This ensures that the order of elements in the set does not affect the resulting tuple,
    making it suitable for use as a dictionary key.

    Parameters:
    - s: The set of strings to convert.

    Returns:
    - A tuple containing the sorted elements from the set.
    """
    return tuple(sorted(s))



# self状态:互斥状态映射
mutually_exclusive_states = {
    'IsLeftHandEmpty': 'IsLeftHolding',
    'IsLeftHolding': 'IsLeftHandEmpty',
    'IsRightHandEmpty': 'IsRightHolding',
    'IsRightHolding': 'IsRightHandEmpty',

    'IsSitting': 'IsStanding',
    'IsStanding': 'IsSitting',

}

# 物体状态: Mapping from state to anti-state
state_to_opposite = {
    'IsOpen': 'IsClose',
    'IsClose': 'IsOpen',
    'IsSwitchedOff': 'IsSwitchedOn',
    'IsSwitchedOn': 'IsSwitchedOff',
    'IsPlugged': 'IsUnplugged',
    'IsUnplugged': 'IsPlugged',
}


def extract_argument(state):
    match = re.search(r'\((.*?)\)', state)
    if match:
        return match.group(1)
    return None


def update_state(c, state_dic):
    for state, opposite in state_to_opposite.items():
        if state in c:
            obj = extract_argument(c)
            if obj in state_dic and opposite in state_dic[obj]:
                return False
            # 更新状态字典
            elif obj in state_dic:
                state_dic[obj].add(state)
            else:
                state_dic[obj] = set()
                state_dic[obj].add(state)
            break
    return True


def check_conflict_RW(c):
    have_at = False
    for str in c:
        if 'Not' not in str and 'RobotNear' in str:
            if have_at:
                return True
            have_at = True

    Holding = False
    HoldingNothing = False
    for str in c:
        if 'Not ' not in str and 'Holding(Nothing)' in str: # 注意 'Not ' in 'Nothing'
            HoldingNothing = True
        if 'Not' not in str and 'Holding(Nothing)' not in str and 'Holding' in str:
            if Holding:
                return True
            Holding = True
        if HoldingNothing and Holding:
            return True
    return False

def check_conflict(conds):

    # conflict = check_conflict_RW(conds)
    # if conflict:
    #     return True


    obj_state_dic = {}
    self_state_dic = {}
    self_state_dic['self'] = set()
    is_near = False
    for c in conds:
        if "IsNear" in c and is_near:
            return True
        elif "IsNear" in c:
            is_near = True
            continue
        # Cannot be updated, the value already exists in the past
        if not update_state(c, obj_state_dic):
            return True
        # Check for mutually exclusive states without obj
        for state, opposite in mutually_exclusive_states.items():
            if state in c and opposite in self_state_dic['self']:
                return True
            elif state in c:
                self_state_dic['self'].add(state)
                break
    # 检查是否同时具有 'IsHoldingCleaningTool(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'
    required_states = {'IsHoldingCleaningTool(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'}
    if all(state in conds for state in required_states):
        return True
    required_states = {'IsHoldingKnife(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'}
    if all(state in conds for state in required_states):
        return True

    return False








def execute_bt(bt,goal, state, verbose=True):
    from btpg.algos.bt_planning.tools import state_transition
    steps = 0
    current_cost = 0
    current_tick_time = 0
    act_num = 1
    record_act = []
    error = False

    val, obj, cost, tick_time = bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
    if verbose:
        print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
    record_act.append(obj.__str__())
    current_tick_time += tick_time
    current_cost += cost
    while val != 'success' and val != 'failure':
        state = state_transition(state, obj)
        val, obj, cost, tick_time = bt.cost_tick(state, 0, 0)
        act_num += 1
        if verbose:
            print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
        record_act.append(obj.__str__())
        current_cost += cost
        current_tick_time += tick_time
        if (val == 'failure'):
            if verbose:
                print("bt fails at step", steps)
            error = True
            break
        steps += 1
        if (steps >= 500):  # 至多运行500步
            break
    if goal <= state:  # 错误解，目标条件不在执行后状态满足
        if verbose:
            print("Finished!")
    else:
        error = True
    # if verbose:
    #     print(f"一定运行了 {act_num-1} 个动作步")
    #     print("current_cost:",current_cost)
    return error, state, act_num - 1, current_cost, record_act[:-1]



def dfs_btml_indent(btml_string, parnode, level=0, is_root=False, act_bt_tree=False):
    indent = " " * (level * 4)  # 4 spaces per indent level
    for child in parnode.children:
        if isinstance(child, Leaf):

            if is_root and len(child.content) > 1:
                # 把多个 cond 串起来
                btml_string += " " * (level * 4) + "sequence\n"
                if act_bt_tree == False:
                    for c in child.content:
                        btml_string += " " * ((level + 1) * 4) + "cond " + str(c) + "\n"

            elif child.type == 'cond':
                # 直接添加cond及其内容，不需要特别处理根节点下多个cond的情况
                # self.btml_string += indent + "cond " + ', '.join(map(str, child.content)) + "\n"
                # 对每个条件独立添加，确保它们各占一行
                if act_bt_tree == False:
                    for c in child.content:
                        btml_string += indent + "cond " + str(c) + "\n"
            elif child.type == 'act':
                # 直接添加act及其内容
                btml_string += indent + 'act ' + child.content.name + "\n"
        elif isinstance(child, ControlBT):
            if child.type == '?':
                btml_string += indent + "selector\n"
                dfs_btml_indent(btml_string,child, level + 1, act_bt_tree=act_bt_tree)  # 增加缩进级别
            elif child.type == '>':
                btml_string += indent + "sequence\n"
                dfs_btml_indent(btml_string, child, level + 1, act_bt_tree=act_bt_tree)  # 增加缩进级别
        return btml_string
def get_btml(bt, use_braces=True, act_bt_tree=False):
    btml_string = "selector\n"
    btml_string = dfs_btml_indent(btml_string,bt.children[0], 1, is_root=True)
    return btml_string



# Function to calculate the percentage of priority actions in expanded actions
def calculate_priority_percentage(expanded, priority_act_ls):
    count_priority_actions = len( {act_name for act_name in expanded if act_name in priority_act_ls})
    percentage = (count_priority_actions / len(priority_act_ls)) * 100 if expanded else 0
    return percentage


def print_action_data_table(goal,start,actions):
    data = []
    for a in actions:
        data.append([a.name , a.pre , a.add , a.del_set , a.cost])
    data.append(["Goal" ,goal ," " ,"Start" ,start])
    # print(tabulate(data, headers=["Name", "Pre", "Add" ,"Del" ,"Cost"], tablefmt="fancy_grid"))  # grid plain simple github fancy_grid


# 从状态随机生成一个行动
def generate_from_state(act,state,num):
    for i in range(0,num):
        if i in state:
            if random.random() >0.5:
                act.pre.add(i)
                if random.random() >0.5:
                    act.del_set.add(i)
                continue
        if random.random() > 0.5:
            act.add.add(i)
            continue
        if random.random() >0.5:
            act.del_set.add(i)

def print_action(act):
    print (act.pre)
    print(act.add)
    print(act.del_set)

def state_transition(state,action):
    if not action.pre <= state:
        print ('error: action not applicable')
        return state
    new_state=(state | action.add) - action.del_set
    return new_state


#行为树测试代码
# def BTTest(seed=1,literals_num=10,depth=10,iters=10,total_count=1000):
#     print("============= BT Test ==============")
#     random.seed(seed)
#     # 设置生成规划问题集的超参数：文字数、解深度、迭代次数
#     literals_num=literals_num
#     depth = depth
#     iters= iters
#     total_tree_size = []
#     total_action_num = []
#     total_state_num = []
#     total_steps_num=[]
#     #fail_count=0
#     #danger_count=0
#     success_count =0
#     failure_count = 0
#     planning_time_total = 0.0
#     # 实验1000次
#     for count in range (total_count):
#
#         action_num = 1
#
#         # 生成一个规划问题，包括随机的状态和行动，以及目标状态
#         states = []
#         actions = []
#         start = generate_random_state(literals_num)
#         state = start
#         states.append(state)
#         #print (state)
#         for i in range (0,depth):
#             a = Action()
#             generate_from_state(a,state,literals_num)
#             if not a in actions:
#                 a.name = "a"+str(action_num)
#                 action_num+=1
#                 actions.append(a)
#             state = state_transition(state,a)
#             if state in states:
#                 pass
#             else:
#                 states.append(state)
#                 #print(state)
#
#         goal = states[-1]
#         state = start
#         for i in range (0,iters):
#             a = Action()
#             generate_from_state(a,state,literals_num)
#             if not a in actions:
#                 a.name = "a"+str(action_num)
#                 action_num+=1
#                 actions.append(a)
#             state = state_transition(state,a)
#             if state in states:
#                 pass
#             else:
#                 states.append(state)
#             state = random.sample(states,1)[0]
#
#         # 选择测试本文算法btalgorithm，或对比算法weakalgorithm
#         algo = OptBTExpAlgorithm()
#         #algo = Weakalgorithm()
#         start_time = time.time()
#         # print_action_data_table(goal, start, list(actions))
#         if algo.run_algorithm(start, goal, actions):#运行算法，规划后行为树为algo.bt
#             total_tree_size.append( algo.bt.count_size()-1)
#             # algo.print_solution()  # 打印行为树
#         else:
#             print ("error")
#         end_time = time.time()
#         planning_time_total += (end_time-start_time)
#
#         #开始从初始状态运行行为树，测试
#         state=start
#         steps=0
#         val, obj = algo.bt.tick(state)#tick行为树，obj为所运行的行动
#         while val !='success' and val !='failure':#运行直到行为树成功或失败
#             state = state_transition(state,obj)
#             val, obj = algo.bt.tick(state)
#             if(val == 'failure'):
#                 print("bt fails at step",steps)
#             steps+=1
#             if(steps>=500):#至多运行500步
#                 break
#         if not goal <= state:#错误解，目标条件不在执行后状态满足
#             #print ("wrong solution",steps)
#             failure_count+=1
#
#         else:#正确解，满足目标条件
#             #print ("right solution",steps)
#             success_count+=1
#             total_steps_num.append(steps)
#         algo.clear()
#         total_action_num.append(len(actions))
#         total_state_num.append(len(states))
#
#     print ("success:",success_count,"failure:",failure_count)#算法成功和失败次数
#     print("Total Tree Size: mean=",np.mean(total_tree_size), "std=",np.std(total_tree_size, ddof=1))#1000次测试树大小
#     print ("Total Steps Num: mean=",np.mean(total_steps_num),"std=",np.std(total_steps_num,ddof=1))
#     print ("Average number of states:",np.mean(total_state_num))#1000次问题的平均状态数
#     print ("Average number of actions",np.mean(total_action_num))#1000次问题的平均行动数
#     print("Planning Time Total:",planning_time_total,planning_time_total/1000.0)
#     print("============ End BT Test ===========")

    # xiao cai
    # success: 1000 failure: 0
    # Total Tree Size: mean= 35.303 std= 29.71336526001515
    # Total Steps Num: mean= 1.898 std= 0.970844240101644
    # Average number of states: 20.678
    # Average number of actions 20.0
    # Planning Time Total: 0.6280641555786133 0.0006280641555786133

    # our start
    # success: 1000 failure: 0
    # Total Tree Size: mean= 17.945 std= 12.841997192488865
    # Total Steps Num: mean= 1.785 std= 0.8120556843187752
    # Average number of states: 20.678
    # Average number of actions 20.0
    # Planning Time Total: 1.4748523235321045 0.0014748523235321046

    # our
    # success: 1000 failure: 0
    # Total Tree Size: mean= 48.764 std= 20.503626574406358
    # Total Steps Num: mean= 1.785 std= 0.8120556843187752
    # Average number of states: 20.678
    # Average number of actions 20.0
    # Planning Time Total: 3.3271877765655518 0.0033271877765655516

