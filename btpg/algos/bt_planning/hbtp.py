import copy
import time
import random
import heapq
import re
from btpg.algos.base.behaviour_tree import Leaf, ControlBT
from btpg.algos.base.planning_action import PlanningAction, state_transition
from btpg.algos.base.btp_base import BTPlannerBase
from collections import deque
import random
import numpy as np
import asyncio
from btpg.algos.llm_client.llms.gpt3 import LLMGPT3
from btpg.algos.llm_client.tools import goal_transfer_str, act_str_process

from btpg.algos.base.tools  import calculate_priority_percentage
from btpg.algos.base.tools  import *
seed = 0
random.seed(seed)
np.random.seed(seed)



class HBTP(BTPlannerBase):
    def __init__(self, verbose=False, llm_reflect=False, llm=None, messages=None, priority_act_ls=None, time_limit=None, \
                 consider_priopity=False, heuristic_choice=-1,output_just_best=True,exp=False,exp_cost=False,
                 theory_priority_act_ls=None):
        self.bt = None
        self.start = None
        self.goal = None
        self.actions = None
        self.min_cost = float('inf')

        self.nodes = []
        self.cycles = 0
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_act =[] # 0602
        self.expanded_percentages = []

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_act = []
        self.traversed_percentages = []
        self.traversed_state_num = 0

        self.bt_without_merge = None
        self.subtree_count = 1

        self.verbose = verbose
        self.bt_merge = True
        self.output_just_best = output_just_best
        self.merge_time = 5

        self.act_bt = None

        self.llm_reflect = llm_reflect
        self.llm = llm
        self.messages = messages

        self.priority_act_ls = priority_act_ls
        if theory_priority_act_ls!=None:
            self.theory_priority_act_ls = theory_priority_act_ls
        else:
            self.theory_priority_act_ls = priority_act_ls

        self.act_cost_dic = {}
        self.time_limit_exceeded = False
        self.time_limit = time_limit

        self.consider_priopity = consider_priopity
        self.heuristic_choice = heuristic_choice

        # 0602
        self.expanded_percentages = []
        self.exp = exp
        self.exp_cost = exp_cost
        self.simu_cost_ls = []
        self.expanded_act_ls_ls = []

    def clear(self):
        self.bt = None
        self.start = None
        self.goal = None
        self.actions = None
        self.min_cost = float('inf')

        self.nodes = []
        self.cycles = 0
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_act =[] # 0602
        self.expanded_percentages = []

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_act = []
        self.traversed_percentages = []
        self.traversed_state_num = 0

        self.bt_without_merge = None
        self.subtree_count = 1

        self.act_bt = None

        # 0602
        self.expanded_percentages = []
        self.simu_cost_ls = []
        self.expanded_act_ls_ls=[]

    def post_processing(self, pair_node, g_cond_anc_pair, subtree, bt, child_to_parent, cond_to_condActSeq):
        '''
        Process the summary work after the algorithm ends.
        '''
        if self.output_just_best:
            # Only output the best
            output_stack = []
            tmp_pair = pair_node
            while tmp_pair != g_cond_anc_pair:
                tmp_seq_struct = cond_to_condActSeq[tmp_pair]
                output_stack.append(tmp_seq_struct)
                tmp_pair = child_to_parent[tmp_pair]

            while output_stack != []:
                tmp_seq_struct = output_stack.pop()
                # print(tmp_seq_struct)
                subtree.add_child([copy.deepcopy(tmp_seq_struct)])

        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt
        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=self.merge_time)
        return bt

    def transfer_pair_node_to_bt(self, path_nodes, root_pair):
        bt = ControlBT(type='cond')

        goal_condition_node = root_pair.cond_leaf
        goal_action_node = root_pair.act_leaf
        bt.add_child([goal_condition_node])

        # subtree = ControlBT(type='?')
        # # subtree.add_child([copy.deepcopy(goal_condition_node),copy.deepcopy(goal_action_node)])
        # subtree.add_child([copy.deepcopy(goal_condition_node)])
        # bt.add_child([subtree])

        queue = deque([root_pair])
        while queue:
            current = queue.popleft()

            # if current.cond_leaf.content != goal_condition_node.content:
            # 建树
            subtree = ControlBT(type='?')
            # subtree.add_child([copy.deepcopy(current.cond_leaf)])
            subtree.add_child([copy.deepcopy(current.cond_leaf)])

            # 过滤掉不在path_nodes中的子节点
            for child in current.children:
                if child not in path_nodes:
                    continue
                # 将过滤后的子节点加入队列
                queue.append(child)

                seq = ControlBT(type='>')
                # seq.add_child([copy.deepcopy(child.cond_leaf), copy.deepcopy(child.act_leaf)])
                seq.add_child([child.cond_leaf, child.act_leaf])
                subtree.add_child([seq])

            parent_of_c = current.cond_leaf.parent
            parent_of_c.children[0] = subtree
        return bt

    def transfer_pair_node_to_act_tree(self, path_nodes, root_pair):
        # 初始化输出字符串，首先添加根节点
        # 将集合中的条件转换为逗号分隔的字符串
        conditions = ', '.join(root_pair.cond_leaf.content)
        # 初始化输出字符串，首先添加根节点和它的条件
        act_tree_string = f'GOAL {conditions}\n'

        # 内部递归函数，用于构建每个节点及其子节点的输出字符串
        def build_act_tree(node, indent, act_count):
            # 存储这个层级生成的字符串，使用序号来标识动作
            node_string = ''
            current_act = 1  # 当前动作的编号，用于生成 ACT 1: 等标签
            for child in node.children:
                if child in path_nodes:
                    # 格式化当前行动的文本
                    prefix = '    ' * indent  # 根据缩进级别生成前缀空格
                    act_label = f'ACT {act_count}.{current_act}: ' if act_count else f'ACT {current_act}: '
                    # 添加当前行动
                    node_string += f'{prefix}{act_label}{child.act_leaf.content.name}  f={round(child.act_leaf.min_cost, 1)} g={round(child.act_leaf.trust_cost, 1)}\n'
                    # 递归添加子行动
                    node_string += build_act_tree(child, indent + 1,
                                                  f'{act_count}.{current_act}' if act_count else str(current_act))
                    current_act += 1  # 更新行动编号

            return node_string

        # 调用递归函数，从根节点的孩子开始，缩进级别为1，活动编号为空字符串
        act_tree_string += build_act_tree(root_pair, 1, '')
        return act_tree_string

    # 调用大模型进行反馈
    def call_large_model(self, goal_cond_act_pair):
        # =================================================
        # 在这里询问大模型，然后更改 act 的值，同时也更新所有 self.nodes 中的值
        # 这里输出前5个cost最长的路径
        # top_five_leaves = heapq.nlargest(5, self.nodes)
        top_five_leaves = heapq.nsmallest(20, self.nodes)
        # 存储路径上的所有结点
        path_nodes = set()
        # 追踪每个叶子结点到根节点的路径
        for leaf in top_five_leaves:
            current = leaf
            while current.parent != None:
                path_nodes.add(current)
                current = current.parent
            path_nodes.add(goal_cond_act_pair)  # 添加根节点
        # for node in path_nodes:
        #     print(node)

        # 构建新的树 动作BT （父节点关系）
        # self.act_bt = self.transfer_pair_node_to_bt(path_nodes=path_nodes,root_pair =goal_cond_act_pair)
        # act_bt_btml_string = self.ACT_BT_get_btml()
        # print(act_bt_btml_string)

        # 如果不建立BT,之间里动作树
        self.act_tree_string = self.transfer_pair_node_to_act_tree(path_nodes=path_nodes, root_pair=goal_cond_act_pair)
        print(self.act_tree_string)
        # =================================================

        prompt = ""
        prompt += self.act_tree_string

        # 大模型返回新的 最优动作，和原来比增加了什么，更新（只更新增加的？）
        # 更新所有动作的值
        # 更新 self.nodes 中所有的cost值？：怎么更新呢，自顶向下bfs遍历更新吗？

        # 先写成同步
        # 这是目前行为树反向扩展算法搜索到的动作树，为达到目标状态，请你在现有动作树的基础上重新推荐接下来达到目标状态还需要的关键动作,
        # 这次不需要输出目标状态，只需要输出关键动作，关键动作的输出格式和此前一样，以 Actions: 开头, 不需要有其它的任何解释问题。
        prompt += (
                "\nThis is the action tree currently found by the reverse expansion algorithm of the behavior tree. " + \
                "To reach the goal state, please recommend the key actions needed next to reach the goal state based on the existing action tree. " + \
                "This time, there is no need to output the goal state, only the key actions are needed. " + \
                'The format for presenting key actions should start with the word \'Actions:\'. ')
        # 'Connect predicates and objects with underscores, and do not use parentheses, brackets, or include any additional explanations or questions.'+\
        # 'For example: Actions: RightGrab_cake, Walk_oven')

        # self.messages.append({"role": "user", "content": prompt})
        # answer = self.llm.request(message=self.messages)
        # self.messages.append({"role": "assistant", "content": answer})
        # print("answer:",answer)
        #
        # act_str = answer.split("Actions:")[1]
        # # act_str = re.sub(r'\s+|[\[\]\(\)\n]', '', act_str)
        # # priority_act_ls = act_str_process(act_str)
        #
        # priority_act_ls = [action.replace(" ", "") for action in act_str.split(",")]
        #
        # print(priority_act_ls)

        # print("before:", heapq.nsmallest(1, self.nodes)[0].act_leaf.content.name)
        # # 重新更新动作序列？和 self.nodes 中的值
        # for act in self.actions:
        #     # act.cost = act.real_cost
        #     if act.name in priority_act_ls:
        #         act.cost = 0
        #
        # temp_nodes = []
        # for node in self.nodes:
        #     node.act_leaf.min_cost = node.act_leaf.parent_cost + node.act_leaf.content.cost
        #     node.cond_leaf.min_cost = node.cond_leaf.parent_cost + node.act_leaf.content.cost
        #     # 重新排序堆，保持最小堆的性质
        #     heapq.heappush(temp_nodes, node)
        # self.nodes = temp_nodes.copy()
        # print("after:", heapq.nsmallest(1, self.nodes)[0].act_leaf.content.name)

    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        '''
        Run the planning algorithm to calculate a behavior tree from the initial state, goal state, and available actions
        '''

        start_time = time.time()

        self.start = start
        self.goal = goal
        self.actions = actions
        self.merge_time = merge_time
        self.traversed_state_num = 0
        min_cost = float('inf')

        child_to_parent = {}
        cond_to_condActSeq = {}

        self.nodes = []
        self.cycles = 0
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_act = []
        self.expanded_percentages = []
        self.expanded_act_ls_ls = []


        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_act = []
        self.traversed_percentages = []

        self.traversed_state_num = 0

        self.bt_without_merge = None
        self.subtree_count = 1
        self.simu_cost_ls = []

        if self.verbose:
            print("\nAlgorithm starts！")

        # 初始化

        for act in self.actions:
            self.act_cost_dic[act.name] = act.cost

        # Initialize the behavior tree with only the target conditions
        bt = ControlBT(type='cond')
        # goal_condition_node = Leaf(type='cond', content=goal, min_cost=0)
        # goal_action_node = Leaf(type='act', content=None, min_cost=0)

        goal_condition_node = Leaf(type='cond', content=goal, min_cost=0)
        goal_action_node = Leaf(type='act', content=None, min_cost=0)

        # Retain the expanded nodes in the subtree first
        subtree = ControlBT(type='?')
        subtree.add_child([copy.deepcopy(goal_condition_node)])
        bt.add_child([subtree])
        goal_cond_act_pair = CondActPair(cond_leaf=goal_condition_node, act_leaf=goal_action_node)

        # I(C,act)
        for act in self.priority_act_ls:
            if act not in goal_cond_act_pair.pact_dic:
                goal_cond_act_pair.pact_dic[act] = 1
            else:
                goal_cond_act_pair.pact_dic[act] += 1

        D_first_cost = 0
        D_first_num = 0
        for key, value in goal_cond_act_pair.pact_dic.items():
            # print(key, value)
            # if key not in self.act_cost_dic.keys():
            #     print("key",key)
            D_first_cost += self.act_cost_dic[key] * value
            D_first_num += value
        goal_condition_node.trust_cost = 0
        goal_action_node.trust_cost = 0
        goal_condition_node.min_cost = D_first_cost
        goal_action_node.min_cost = D_first_cost

        # Using priority queues to store extended nodes
        heapq.heappush(self.nodes, goal_cond_act_pair)
        # self.expanded.append(goal)
        self.expanded.append(goal_condition_node)
        self.traversed_state_num += 1
        self.traversed = [goal]  # Set of expanded conditions

        if goal <= start:
            self.bt_without_merge = bt
            print("goal <= start, no need to generate bt.")
            return bt, 0, self.time_limit_exceeded

        epsh = 0
        while len(self.nodes) != 0:

            # 0602 记录有多少动作在里面了
            # print("self.priority_act_ls",self.priority_act_ls)
            # 当前是第 len(self.expanded) 个
            # 求对应的扩展的动作里占了self.priority_act_ls的百分之几
            # Add the initial percentage for the goal node
            if self.exp:
                self.expanded_act_ls_ls.append(self.expanded_act)
                self.expanded_percentages.append(calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                self.traversed_percentages.append(calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))



            # 调用大模型
            if self.llm_reflect:
                # if len(self.expanded) % 2000 == 0 and len(self.expanded) >= 100:
                #     print(len(self.expanded))
                # if len(self.expanded) % 1000 == 0 and len(self.expanded)>=100:
                #      self.call_large_model(goal_cond_act_pair=goal_cond_act_pair)
                if len(self.expanded) >= 1:
                    self.call_large_model(goal_cond_act_pair=goal_cond_act_pair)

            self.cycles += 1
            #  Find the condition for the shortest cost path
            # min_cost = float('inf')
            current_pair = heapq.heappop(self.nodes)
            min_cost = current_pair.cond_leaf.min_cost

            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            c = current_pair.cond_leaf.content

            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(  # 这里做 ACT TREE 时候，没有copy 被更新了父节点
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])
                # self.expanded.append(c)
                self.expanded.append(current_pair.cond_leaf)
                self.expanded_act.append(current_pair.act_leaf.content.name)

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])

                if c <= start:
                    bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                              cond_to_condActSeq)
                    if self.exp:
                        self.expanded_act_ls_ls.append(self.expanded_act)
                        self.expanded_percentages.append(
                            calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                        self.traversed_percentages.append(
                            calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))

                    return bt, min_cost, self.time_limit_exceeded
            # =============额外家的
            elif c == set() and c <= start:
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(  # 这里做 ACT TREE 时候，没有copy 被更新了父节点
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])
                # self.expanded.append(c)
                self.expanded.append(current_pair.cond_leaf)
                self.expanded_act.append(current_pair.act_leaf.content.name)

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                return bt, min_cost, self.time_limit_exceeded
            # =============额外家的
            # 超时处理
            if self.time_limit != None and time.time() - start_time > self.time_limit:
                self.time_limit_exceeded = True
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                return bt, min_cost, self.time_limit_exceeded

                if self.verbose:
                    print("Expansion complete for action node={}, with new conditions={}, min_cost={}".format(
                        current_pair.act_leaf.content.name, current_pair.cond_leaf.content,
                        current_pair.cond_leaf.min_cost))

            if self.verbose:
                print("Traverse all actions and find actions that meet the conditions:")
                print("============")
            current_mincost = current_pair.cond_leaf.min_cost
            current_trust = current_pair.cond_leaf.trust_cost


            if self.exp_cost:
            # 模拟调用计算cost
                tmp_bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                error, state, act_num, current_cost, record_act_ls = execute_bt(tmp_bt,goal, c,
                                                                                 verbose=False)
                self.simu_cost_ls.append(current_cost)


            # if self.verbose:
            # if current_pair.act_leaf.content != None:
            #     print("current act:", current_pair.act_leaf.content.name)
            #     print("current cond:", c)
            #     print("cost:", current_pair.cond_leaf.min_cost)

            # ====================== Action Trasvers ============================ #
            # Traverse actions to find applicable ones
            traversed_current = []
            for act in actions:
                # if "Turn" in act.name:
                #     tt = 1

                epsh += 0.00000000001
                if not c & ((act.pre | act.add) - act.del_set) <= set():
                    if (c - act.del_set) == c:
                        # if self.verbose:
                        #     # Action satisfies conditions for expansion
                        #     print(f"———— 动作：{act.name}  满足条件可以扩展")
                        c_attr = (act.pre | c) - act.add

                        if check_conflict(c_attr):
                            # if self.verbose:
                            #     print("———— Conflict: action={}, conditions={}".format(act.name, act))
                            continue

                        # 剪枝操作,现在的条件是以前扩展过的条件的超集
                        valid = True

                        if self.consider_priopity:
                            for expanded_condition in self.expanded:
                                if expanded_condition.content <= c_attr:
                                    # 考虑动作优先级的时候\启发式,剪枝 还有额外的条件
                                    # 可以剪枝的时候判断一些路径
                                    #  cur_g >= g
                                    # if current_trust + act.cost >= expanded_condition.min_cost:
                                    valid = False
                                    break
                            pass
                        else:
                            # 不考虑动作优先级的时候\启发式,直接剪枝
                            for expanded_condition in self.expanded:
                                if expanded_condition.content <= c_attr:
                                    valid = False
                                    break

                        if valid:

                            # g=trust_cost
                            # h= h_cost
                            # c_attr_node = Leaf(type='cond', content=c_attr, trust_cost=current_trust + act.cost)
                            # a_attr_node = Leaf(type='act', content=act, trust_cost=current_trust + act.cost)

                            # c_attr_node = Leaf(type='cond', content=c_attr, parent_cost=current_mincost)
                            # a_attr_node = Leaf(type='act', content=act, parent_cost=current_mincost)

                            c_attr_node = Leaf(type='cond', content=c_attr, min_cost=current_mincost + act.cost,
                                               parent_cost=current_mincost)
                            a_attr_node = Leaf(type='act', content=act, min_cost=current_mincost + act.cost,
                                               parent_cost=current_mincost)

                            new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)
                            new_pair.path = current_pair.path + 1

                            # I(C,act)
                            # new_cost = current_mincost + act.cost
                            # new_pair.pact_dic = copy.deepcopy(current_pair.pact_dic)
                            # if act.name in new_pair.pact_dic and new_pair.pact_dic[act.name] > 0:
                            #     new_cost = current_mincost + act.priority
                            #     new_pair.pact_dic[act.name] -= 1
                            # c_attr_node.min_cost = new_cost
                            # a_attr_node.min_cost = new_cost

                            h = 0

                            new_pair.pact_dic = copy.deepcopy(current_pair.pact_dic)
                            if act.name in new_pair.pact_dic and new_pair.pact_dic[act.name] > 0:
                                new_pair.pact_dic[act.name] -= 1
                                c_attr_node.min_cost = current_mincost + act.priority
                                a_attr_node.min_cost = current_mincost + act.priority

                            # 最迟启发式：h 是到重点的距离，通过计算 new_pair.pact_dic得到
                            # h = 0
                            # new_pair.pact_dic = copy.deepcopy(current_pair.pact_dic)
                            # if act.name in new_pair.pact_dic and new_pair.pact_dic[act.name] > 0:
                            #     new_pair.pact_dic[act.name] -= 1
                            #
                            # for key, value in new_pair.pact_dic.items():
                            #     # print(key, value)
                            #     h += self.act_cost_dic[key] * value
                            # c_attr_node.min_cost = c_attr_node.trust_cost + h - epsh
                            # a_attr_node.min_cost = a_attr_node.trust_cost + h - epsh

                            # h = 0
                            # new_pair.pact_dic = copy.deepcopy(current_pair.pact_dic)
                            # if act.name in new_pair.pact_dic and new_pair.pact_dic[act.name] > 0:
                            #     new_pair.pact_dic[act.name] -= 1
                            #
                            # for key, value in new_pair.pact_dic.items():
                            #     # print(key, value)
                            #     h += self.act_cost_dic[key] * value
                            # c_attr_node.min_cost = c_attr_node.trust_cost + h*10000 - epsh
                            # a_attr_node.min_cost = a_attr_node.trust_cost + h*10000 - epsh

                            # 启发式：乘一下
                            # new_pair.pact_dic = copy.deepcopy(current_pair.pact_dic)
                            # if act.name in new_pair.pact_dic and new_pair.pact_dic[act.name] > 0:
                            #     new_pair.pact_dic[act.name] -= 1
                            # remaining = 0
                            # remaining_num = 0
                            # g = current_trust + act.cost
                            # for key, value in new_pair.pact_dic.items():
                            #     # print(key, value)
                            #     remaining += self.act_cost_dic[key] * value
                            #     remaining_num += value
                            # h =remaining * (remaining_num / D_first_num)
                            # # h = min(max(0,D_first_cost-g), remaining * (remaining_num / D_first_num))
                            # c_attr_node.min_cost = c_attr_node.trust_cost + h - epsh
                            # a_attr_node.min_cost = a_attr_node.trust_cost + h - epsh

                            # 0425启发式：h=max(0,D-g)* (剩余/D)
                            # g=current_trust + act.cost
                            # # max(0,D_first_cost-g)
                            # remaining = 0
                            # for key, value in new_pair.pact_dic.items():
                            #     remaining+=self.act_cost_dic[key] * value
                            # if D_first_cost==0:
                            #     h=0
                            # else:
                            #     h=max(0,D_first_cost-g)*(remaining/D_first_cost)
                            # c_attr_node.min_cost = c_attr_node.trust_cost + h - epsh
                            # a_attr_node.min_cost = a_attr_node.trust_cost + h - epsh

                            heapq.heappush(self.nodes, new_pair)

                            # 记录结点的父子关系
                            new_pair.parent = current_pair
                            current_pair.children.append(new_pair)

                            # 如果之前标记过/之前没标记但现在是1
                            if (
                                    current_pair.isCostOneAdded == False and act.cost == 1) or current_pair.isCostOneAdded == True:
                                new_pair.isCostOneAdded = True
                            # 之前标记过但现在是1 表示多加了1
                            if current_pair.isCostOneAdded == True and act.cost == 1:
                                new_pair.cond_leaf.min_cost -= 1
                                new_pair.act_leaf.min_cost -= 1

                            # Need to record: The upper level of c_attr is c
                            if self.output_just_best:
                                child_to_parent[new_pair] = current_pair

                            self.traversed_state_num += 1
                            # Put all action nodes that meet the conditions into the list
                            traversed_current.append(c_attr)
                            self.traversed_act.append(act.name)

                            if self.verbose:
                                print("———— -- Action={} meets conditions, new condition={}".format(act.name, c_attr))

            self.traversed.extend(traversed_current)
            # ====================== End Action Trasvers ============================ #

        bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                  cond_to_condActSeq)

        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt

        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=merge_time)

        if self.verbose:
            print("Error: Couldn't find successful bt!")
            print("Algorithm ends!\n")

        return bt, min_cost, self.time_limit_exceeded


    def run_algorithm_test(self, start, goal, actions):
        self.bt, mincost = self.run_algorithm_selTree(start, goal, actions)
        return True

