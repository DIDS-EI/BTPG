import re
import os
from btpg.utils import ROOT_PATH
os.chdir(f'{ROOT_PATH}/../test_examples')
import random
import numpy as np
seed = 0
random.seed(seed)
np.random.seed(seed)

from btpg.algos.base.tools import get_btml
from btpg.algos.bt_planning.examples import *
from btpg.algos.base.planning_action import state_transition

from btpg.algos.bt_planning.bt_expansion import BTExpansion
from btpg.algos.bt_planning.reactive_planner import ReactivePlanning
from btpg.algos.bt_planning.obtea import OBTEA
from btpg.algos.bt_planning.hbtp import HBTP
from btpg.algos.bt_planning.uhbtp import UHBTP


# Used for experimental test data recording
from btpg.algos.bt_planning_exp.bt_expansion_test import BTExpansionTest
from btpg.algos.bt_planning_exp.obtea_test import OBTEATest
from btpg.algos.bt_planning_exp.hbtp_test import HBTPTest
from btpg.algos.bt_planning_exp.uhbtp_test import UHBTPTest
from metric.Execution_Robustnes.tools import modify_condition_set_Random_Perturbations


# 封装好的主接口
class BTPlannerInterface:
    def __init__(self, behavior_lib, cur_cond_set, priority_act_ls=[], key_predicates=[], key_objects=[], selected_algorithm="opt",
                 mode="big",
                 bt_algo_opt=True, llm_reflect=False, llm=None, messages=None, action_list=None,use_priority_act=True,time_limit=None,
                 heuristic_choice=-1,output_just_best=True,exp=False,exp_cost=False,max_expanded_num=None,
                 theory_priority_act_ls=None):
        """
        Initialize the BTOptExpansion with a list of actions.
        :param action_list: A list of actions to be used in the behavior tree.
        """


        self.cur_cond_set = cur_cond_set
        self.bt_algo_opt = bt_algo_opt
        self.selected_algorithm = selected_algorithm
        self.time_limit=time_limit

        self.output_just_best = output_just_best
        self.exp = exp
        self.exp_cost = exp_cost
        self.max_expanded_num=max_expanded_num

        # 剪枝操作,现在的条件是以前扩展过的条件的超集
        self.consider_priopity = False

        # 选择全是0的启发式，还是 cost/10000 的启发式，还是都不采用
        # 定义变量 heuristic_choice：
        # 0 表示全是 0 的启发式
        # 1 表示 cost/10000 的启发式
        # -1 表示不使用启发式
        self.heuristic_choice = heuristic_choice  # 可以根据需要更改这个值

        # 自定义动作空间
        if behavior_lib == None:
            self.actions = action_list
            self.big_actions=self.actions
        # 默认的大动作空间
        else:
            self.big_actions = collect_action_nodes(behavior_lib)


        if mode == "big":
            self.actions = self.big_actions
        elif mode=="user-defined":
            self.actions = action_list
            # print(f"自定义小动作空间：收集到 {len(self.actions)} 个动作")
            # print("----------------------------------------------")
        elif mode=="small-objs":
            self.actions = self.collect_compact_object_actions(key_objects)
            # print(f"选择小动作空间，只考虑物体：收集到 {len(self.actions)} 个动作")
            # print("----------------------------------------------")
        elif mode=="small-predicate-objs":
            self.actions = self.collect_compact_predicate_object_actions(key_predicates,key_objects)
            # print(f"选择小动作空间，考虑谓词和物体：收集到 {len(self.actions)} 个动作")
            # print("----------------------------------------------")
        else:
            raise ValueError(f"Invalid mode: {mode}, please select from 'big', 'user-defined', 'small-objs', 'small-predicate-objs'")
        
        if use_priority_act:
            self.priority_act_ls = self.filter_actions(priority_act_ls)
            self.priority_obj_ls = key_objects
        else:
            self.priority_act_ls=[]
            self.priority_obj_ls =[]
        if self.priority_act_ls !=[]:
            self.consider_priopity=True


        # if self.heuristic_choice==-1: 在 adjust_action_priority 里面已经写了这个控制
        #     self.priority_act_ls=[]

        if theory_priority_act_ls!=None:
            self.theory_priority_act_ls=theory_priority_act_ls
        else:
            self.theory_priority_act_ls=self.priority_act_ls

        self.actions = self.adjust_action_priority(self.actions, self.priority_act_ls, self.priority_obj_ls,
                                                       self.selected_algorithm)

        self.has_processed = False
        self.llm_reflect = llm_reflect
        self.llm = llm
        self.messages = messages

        self.min_cost = float("inf")

    def process(self, goal):
        """
        Process the input sets and return a string result.
        :param input_set: The set of goal states and the set of initial states.
        :return: A btml string representing the outcome of the behavior tree.
        """
        self.goal = goal
        if not self.exp_cost:
            if self.selected_algorithm == "opt":
                self.algo = HBTP(verbose=False, \
                                              llm_reflect=self.llm_reflect, llm=self.llm, messages=self.messages, \
                                              priority_act_ls=self.priority_act_ls,time_limit=self.time_limit,
                                              consider_priopity = self.consider_priopity,exp_cost=self.exp_cost,
                                              heuristic_choice = self.heuristic_choice,output_just_best=self.output_just_best,
                                              exp=self.exp,theory_priority_act_ls=self.theory_priority_act_ls)
            elif self.selected_algorithm == "obtea":
                self.algo = OBTEA(verbose=False, \
                                              llm_reflect=self.llm_reflect, llm=self.llm, messages=self.messages, \
                                              priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                              consider_priopity=self.consider_priopity,exp_cost=self.exp_cost,
                                              heuristic_choice=self.heuristic_choice,output_just_best=self.output_just_best,
                                          exp=self.exp,theory_priority_act_ls=self.theory_priority_act_ls)
            elif self.selected_algorithm == "bfs":
                self.algo = BTExpansion(verbose=False,time_limit = self.time_limit,output_just_best=self.output_just_best,
                                           priority_act_ls=self.priority_act_ls,exp_cost=self.exp_cost,exp=self.exp,theory_priority_act_ls=self.theory_priority_act_ls)
                # self.algo = BTalgorithm(verbose=False)
            # elif self.selected_algorithm == "dfs":
            #     self.algo = BTalgorithmDFS(verbose=False)
            elif self.selected_algorithm == "weak":
                self.algo = ReactivePlanning(verbose=False,time_limit = self.time_limit,output_just_best=self.output_just_best,priority_act_ls=self.priority_act_ls,exp=self.exp)
            elif self.selected_algorithm == "hbtp":
                info_dict = {
                    'priority_act_ls': [],
                    'arg_set': {'solution', 'delete','value','robust'},
                                        'priority_cond_set': set(),
                                      'max_expanded_num':self.max_expanded_num
                }
                self.algo = UHBTP(verbose=False, act_tree_verbose=False,
                             priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                             output_just_best=self.output_just_best,exp_record=self.exp,exp_cost=self.exp_cost,exp=self.exp,
                                 continue_expand=False,max_expanded_num=self.max_expanded_num,
                                                    theory_priority_act_ls=self.theory_priority_act_ls,info_dict=info_dict)
            else:
                print("Error in algorithm selection: This algorithm does not exist.")
        else:
            if self.selected_algorithm == "opt":
                self.algo = HBTPTest(verbose=False, \
                                              llm_reflect=self.llm_reflect, llm=self.llm, messages=self.messages, \
                                              priority_act_ls=self.priority_act_ls,time_limit=self.time_limit,
                                              consider_priopity = self.consider_priopity,exp_cost=self.exp_cost,
                                              heuristic_choice = self.heuristic_choice,output_just_best=self.output_just_best,\
                                                   exp=self.exp,max_expanded_num=self.max_expanded_num)
            elif self.selected_algorithm == "obtea":
                self.algo = OBTEATest(verbose=False, \
                                              llm_reflect=self.llm_reflect, llm=self.llm, messages=self.messages, \
                                              priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                              consider_priopity=self.consider_priopity,exp_cost=self.exp_cost,
                                              heuristic_choice=self.heuristic_choice,output_just_best=self.output_just_best,\
                                               exp=self.exp,max_expanded_num=self.max_expanded_num)
            elif self.selected_algorithm == "bfs":
                self.algo = BTExpansionTest(verbose=False,time_limit = self.time_limit,output_just_best=self.output_just_best,\
                                                priority_act_ls=self.priority_act_ls,exp_cost=self.exp_cost,exp=self.exp,\
                                                max_expanded_num=self.max_expanded_num)
            elif self.selected_algorithm == "hbtp":
                info_dict = {
                    'priority_act_ls': [],
                    'arg_set': {'solution', 'delete','value','robust'},
                                        'priority_cond_set': set(),
                                      'max_expanded_num':self.max_expanded_num
                }
                self.algo = UHBTPTest(verbose=False, act_tree_verbose=False,
                             priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                             output_just_best=self.output_just_best,exp_record=self.exp,exp_cost=self.exp_cost,exp=self.exp,
                                 continue_expand=False,max_expanded_num=self.max_expanded_num,
                                                    theory_priority_act_ls=self.theory_priority_act_ls,info_dict=info_dict)
            else:
                print("Error in algorithm selection: This algorithm does not exist.")
            # elif self.selected_algorithm == "baseline":
            #     self.algo = OptBTExpAlgorithm_BaseLine(verbose=False, \
            #                                   llm_reflect=self.llm_reflect, llm=self.llm, messages=self.messages, \
            #                                   priority_act_ls=self.priority_act_ls)
            # elif self.selected_algorithm == "opt-h":
            #     self.algo = OptBTExpAlgorithmHeuristics(verbose=False)


        self.algo.clear()
        self.algo.run_algorithm(self.cur_cond_set, self.goal, self.actions)  # 调用算法得到行为树保存至 algo.bt

        # self.btml_string = self.algo.get_btml()
        self.has_processed = True
        # algo.print_solution() # print behavior tree

        # return self.btml_string
        return True

    def post_process(self,ptml_string=True):
        if ptml_string:
            self.btml_string = self.algo.get_btml()
            # self.btml_string = get_btml(self.algo.bt)
        else:
            self.btml_string=""
        if self.selected_algorithm == "opt":
            self.min_cost = self.algo.min_cost
        else:
            self.min_cost = self.algo.get_cost()
        return self.btml_string, self.min_cost, len(self.algo.expanded)

    def filter_actions(self, priority_act_ls):
        # 创建一个集合，包含所有标准动作的名称
        standard_actions_set = {act.name for act in self.actions} # self.big_actions
        # 过滤 priority_act_ls，只保留名称在标准集合中的动作
        filtered_priority_act_ls = [act for act in priority_act_ls if act in standard_actions_set]
        return filtered_priority_act_ls


    def adjust_action_priority(self, action_list, priority_act_ls, priority_obj_ls, selected_algorithm):
        # recommended_acts=["RightPutIn(bananas,fridge)",
        #                   "Open(fridge)",
        #                   "Walk(fridge)",
        #                   "Close(fridge)",
        #                   "RightGrab(bananas)",
        #                   "Walk(bananas)"
        #                   ]

        recommended_acts = priority_act_ls
        recommended_objs = priority_obj_ls

        # for act in action_list:
        #     act.cost = 0

        # if selected_algorithm == "opt-h":
        #     for act in action_list:
        #         if act.name in recommended_acts:
        #             act.priority = 0
        # else:

        # 根据目标中的物体，调整有这些物体的优先级
        # 正则表达式用于找到括号中的内容
        # print("============ Priority Objs: ==============")
        # pattern = re.compile(r'\((.*?)\)')
        # for act in action_list:
        #     match = pattern.search(act.name)
        #     if match:
        #         # 将括号内的内容按逗号分割
        #         action_objects = match.group(1).split(',')
        #         # 遍历每个物体名称
        #         if all(obj in recommended_objs for obj in action_objects):
        #             # act.priority = 0.000001
        #             # act.priority = 1
        #             act.priority = 1
        #             # print(act)
        # print("============ Priority Objs: ==============")

        for act in action_list:
            act.priority = act.cost
            if act.name in recommended_acts:
                if self.heuristic_choice==0:
                    act.priority = 0
                elif self.heuristic_choice==1:
                    act.priority = act.priority * 1.0 / 10000

                # act.cost= act.cost*1.0/10000
                # act.cost=0
                # act.priority = act.cost*1.0/100000

        # 对action排序
        # action_list.sort(key=lambda x: x.priority)
        # action_list.sort(key=lambda x: x.cost)
        # action_list.sort(key=lambda x: (x.priority, x.name))
        # action_list.sort(key=lambda x: (x.priority, -ord(x.name[0])))
        action_list.sort(key=lambda x: (x.priority, x.real_cost, x.name ))

        # for act in action_list:
        #     if act.priority <= 1 :
        #         act.cost = 1000000

        return action_list


    def collect_compact_object_actions(self, key_objects):
        small_act=[]
        pattern = re.compile(r'\((.*?)\)')
        for act in self.big_actions:
            match = pattern.search(act.name)

            # if "Put" in act.name and "apple" in act.name:
            #     print(act.name)
            #     pass

            if match:
                # 将括号内的内容按逗号分割
                action_objects = match.group(1).split(',')
                # 遍历每个物体名称
                if all(obj in key_objects for obj in action_objects):
                    small_act.append(act)
        return small_act

    def collect_compact_predicate_object_actions(self, key_predicates, key_objects):
        small_act = []
        # 正则表达式提取括号内的内容
        pattern = re.compile(r'\((.*?)\)')

        for act in self.big_actions:
            # 使用 `any()` 函数检查 act.name 是否包含任何一个 key_predicates 中的谓词
            if any(predicate in act.name for predicate in key_predicates):
                # 提取括号内的对象列表
                match = pattern.search(act.name)
                if match:
                    action_objects = match.group(1).split(',')
                    # 检查所有对象是否都在 key_objects 中
                    if all(obj in key_objects for obj in action_objects):
                        small_act.append(act)

        return small_act


    def execute_bt(self,goal,state,verbose=True):
        from btpg.algos.base.tools import state_transition
        steps = 0
        current_cost = 0
        current_tick_time = 0
        act_num=1
        record_act = []
        error=False

        val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
        if verbose:
            print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
        record_act.append(obj.__str__())
        current_tick_time += tick_time
        current_cost += cost
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)
            val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)
            act_num+=1
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
            if (steps >= 300):  # 至多运行500步
                break
        if goal <= state:  # 错误解，目标条件不在执行后状态满足
            if verbose:
                print("Finished!")
        else:
            error = True
        # if verbose:
        #     print(f"一定运行了 {act_num-1} 个动作步")
        #     print("current_cost:",current_cost)
        return error,state,act_num-1,current_cost,record_act[:-1],current_tick_time

    def execute_bt_Random_Perturbations(self,scene,SimAct,objects,goal,state, verbose=True, p=0.2):
        from btpg.algos.base.tools import state_transition
        steps = 0
        current_cost = 0
        current_tick_time = 0
        act_num=1
        record_act = []
        error=False

        val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
        if verbose:
            print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
        record_act.append(obj.__str__())
        current_tick_time += tick_time
        current_cost += cost
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)

            state = modify_condition_set_Random_Perturbations(scene,SimAct, state, objects, p=p)

            val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)
            act_num+=1
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
        return error,state,act_num-1,current_cost,record_act[:-1],current_tick_time

    # 方法一：查找所有初始状态是否包含当前状态
    def find_all_leaf_states_contain_start(self, start):
        if not self.has_processed:
            raise RuntimeError("The process method must be called before find_all_leaf_states_contain_start!")
        # 返回所有能到达目标状态的初始状态
        state_leafs = self.algo.get_all_state_leafs()
        for state in state_leafs:
            if start >= state:
                return True
        return False

    # 方法二：模拟跑一遍行为树，看 start 能够通过执行一系列动作到达 goal
    def run_bt_from_start(self, goal, start):
        if not self.has_processed:
            raise RuntimeError("The process method must be called before run_bt_from_start!")
        # 检查是否能到达目标
        right_bt = True
        state = start
        steps = 0
        val, obj = self.algo.bt.tick(state)
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)
            val, obj = self.algo.bt.tick(state)
            if (val == 'failure'):
                # print("bt fails at step", steps)
                right_bt = False
            steps += 1
        if not goal <= state:
            # print("wrong solution", steps)
            right_bt = False
        else:
            pass
            # print("right solution", steps)
        return right_bt


def collect_action_nodes(behavior_lib):
    action_list = []
    can_expand_ored=0
    for cls in behavior_lib["Action"].values():
        if cls.can_be_expanded:
            can_expand_ored+=1
            # print(f"可扩展动作：{cls.__name__}, 存在{len(cls.valid_args)}个有效论域组合")
            # print({cls.__name__})
            if cls.num_args == 0:
                action_list.append(PlanningAction(name=cls.get_ins_name(), **cls.get_info()))
            if cls.num_args == 1:
                for arg in cls.valid_args:
                    action_list.append(PlanningAction(name=cls.get_ins_name(arg), **cls.get_info(arg)))
            if cls.num_args > 1:
                for args in cls.valid_args:
                    action_list.append(PlanningAction(name=cls.get_ins_name(*args), **cls.get_info(*args)))

    # print(f"共收集到 {len(action_list)} 个实例化动作")
    # print(f"共收集到 {can_expand_ored} 个动作谓词")

    # for a in self.action_list:
    #     if "Turn" in a.name:
    #         print(a.name)
    # print("--------------------\n")

    return action_list


def collect_conditions(node):
    from btpg.algos.base.behaviour_tree import Leaf
    conditions = set()
    if isinstance(node, Leaf) and node.type == 'cond':
        # 如果是叶子节点并且类型为'cond'，则将内容添加到集合中
        conditions.update(node.content)
    elif hasattr(node, 'children'):
        # 对于有子节点的控制节点，递归收集所有子节点的条件
        for child in node.children:
            conditions.update(collect_conditions(child))
    return conditions


