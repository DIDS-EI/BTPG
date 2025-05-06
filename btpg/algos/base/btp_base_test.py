import copy
import time
import random
import heapq
import re
from btpg.algos.base.planning_action import PlanningAction
from btpg.algos.base.btp_base import BTPlannerBase,CondActPair
from collections import deque
from btpg.algos.base.tools import *
seed = 0
random.seed(seed)
np.random.seed(seed)



class BTPlannerBaseTest(BTPlannerBase):
    def __init__(self, verbose=False, act_tree_verbose=False,
                 priority_act_ls=None, time_limit=None,\
                 output_just_best=True,bt_merge=True,max_expanded_num=None,exp_cost=False,exp=False,**kwargs):

        self.use_robust = None
        self.bt = None
        self.bt_merge = bt_merge
        self.merge_time = 5
        self.bt_without_merge = None

        self.start = None
        self.goal = None
        self.actions = None
        self.min_cost = float('inf')

        self.nodes = []
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_num=0
        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue


        self.verbose = verbose
        self.output_just_best = output_just_best

        self.act_bt = None
        self.act_tree_verbose = act_tree_verbose


        self.act_cost_dic = {}
        self.time_limit_exceeded = False
        self.time_limit = time_limit

        self.priority_act_ls = priority_act_ls


        self.goal_cond_act_pair = None

        self.calculate_cost = None

        self.traversed_state_num = 0

        self.exp_record = kwargs.get('exp_record',False)
        self.max_expanded_num = kwargs.get('max_expanded_num',None)

        if not self.max_expanded_num:
            self.max_expanded_num = max_expanded_num

        # print("self.max_expanded_num:", self.max_expanded_num)

        self.continue_expand = kwargs.get('continue_expand',False)


        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_act =[] # 0602
        self.expanded_percentages = []

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_act = []
        self.traversed_percentages = []
        self.traversed_state_num = 0
        
        self.exp = exp
        self.exp_cost = exp_cost
        self.max_min_cost_ls = []
        self.simu_cost_ls = []

        self.is_robust_expand = False

    def pre_process(self):
        self.calculate_cost = self.sum_cost


    def clear(self):
        self.bt = None
        self.merge_time = 5
        self.bt_without_merge = None

        self.start = None
        self.goal = None
        self.actions = None
        self.min_cost = float('inf')

        self.nodes = []
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_num=0
        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue

        self.act_bt = None

        self.act_cost_dic = {}
        self.time_limit_exceeded = False
        

        self.expanded_act =[] # 0602
        self.expanded_percentages = []

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_act = []
        self.traversed_percentages = []
        self.traversed_state_num = 0


    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        '''
        Run the planning algorithm to calculate a behavior tree from the initial state, goal state, and available actions
        '''
        start_time = time.time()

        self.start = start
        self.goal = goal
        self.actions = actions
        self.merge_time = merge_time
        
        
        cost_every_exp=0
        cost_act_num_every_exp = 0
        self.simu_cost_act_num_ls=[]
        self.cost_act_num_ratio=[]
        
        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_act =[] # 0602
        self.expanded_percentages = []

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue

        self.traversed_act = []
        self.traversed_percentages = []
        self.traversed_state_num = 0
        
        self.tree_size_ls=[]
        

        self.pre_process()

        min_cost = float('inf')

        child_to_parent = {}
        cond_to_condActSeq = {}

        if self.verbose:
            print("\nAlgorithm starts！")

        # Initialize the behavior tree with only the target conditions
        bt = ControlBT(type='cond')
        goal_condition_node = Leaf(type='cond', content=goal, min_cost=0,trust_cost=0)
        goal_action_node = Leaf(type='act', content=None, min_cost=0,trust_cost=0)
        bt.add_child([goal_condition_node])

        # Retain the expanded nodes in the subtree first
        subtree = ControlBT(type='?')
        subtree.add_child([copy.deepcopy(goal_condition_node)])

        # bt.add_child([subtree])
        goal_cond_act_pair = CondActPair(cond_leaf=goal_condition_node, act_leaf=goal_action_node)

        self.goal_cond_act_pair = goal_cond_act_pair

        # Using priority queues to store extended nodes
        # self.nodes.append(goal_cond_act_pair)
        self.put_pair(goal_cond_act_pair)

        # Using priority queues to store extended nodes
        self.expanded.append(goal)
        self.tree_size_ls.append(goal)
        self.traversed_state_num += 1
        traversed_current = [goal]
        self.traversed_new = [goal]

        if goal <= start and not self.continue_expand:
            self.bt_without_merge = bt
            print("goal <= start, no need to generate bt.")
            return bt, 0,self.time_limit_exceeded

        while len(self.nodes) != 0:
            
            print("len(self.expanded):", len(self.expanded))

            if self.exp :
                self.expanded_percentages.append(calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                self.traversed_percentages.append(calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))

            current_pair = self.pop_pair()
            if current_pair.cond_leaf.content in self.traversed:
                continue


            min_cost = current_pair.cond_leaf.min_cost

            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            c = current_pair.cond_leaf.content

            # experience new
            # if self.exp_record:
            #     if current_pair.act_leaf.content!=None:
            #         self.expanded_act_ls.append(current_pair.act_leaf.content.name)
            #     self.expanded_act_ls_ls.append(self.expanded_act_ls)
            #     self.expanded_percentages_ls.append(calculate_priority_percentage(self.expanded_act_ls, self.theory_priority_act_ls))
            # print("len(self.expanded):", len(self.expanded))
            if self.continue_expand and len(self.expanded)>self.max_expanded_num:
                
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                            cond_to_condActSeq)
                return bt, min_cost, self.time_limit_exceeded

            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                if self.output_just_best:
                    sequence_structure = ControlBT(type='>')
                    sequence_structure.add_child(
                        [current_pair .cond_leaf, current_pair .act_leaf])
                    cond_to_condActSeq[current_pair] = sequence_structure
                    

                subtree = ControlBT(type='?')
                subtree.add_child([copy.deepcopy(current_pair.cond_leaf)])  # 子树首先保留所扩展结点

                self.expanded.append(c)
                self.expanded_act.append(current_pair.act_leaf.content.name)

                # if c <= start and not self.continue_expand:
                #
                #     if not self.continue_expand:
                #         bt = self.post_processing(current_pair , goal_cond_act_pair, subtree, bt,child_to_parent,cond_to_condActSeq)
                #         return bt, min_cost,self.time_limit_exceeded
                #     else:
                #         if self.use_robust:
                #             self.create_robust_literal_value_dict()


                if self.verbose:
                    print("Expansion complete for action node={}, with new conditions={}, min_cost={}".format(
                        current_pair.act_leaf.content.name, current_pair.cond_leaf.content,
                        current_pair.cond_leaf.min_cost))

            if self.verbose:
                print("Traverse all actions and find actions that meet the conditions:")
                print("============")
            current_mincost = current_pair.cond_leaf.min_cost
            current_trust = current_pair.cond_leaf.trust_cost


            # 模拟调用计算cost
            if self.exp_cost:
                # cal  current_cost
                # 一共 self.max_expanded_num
                # self.max_expanded_num=10
                # traversed_current 中全部都需要算一个  cost
                # exp=False,exp_cost=True,output_just_best=False,max_expanded_num=max_epoch
                # cost_every_exp = self.max_expanded_num - len(self.expanded) + 1
                tmp_bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                error, state, act_num, cur_cost, record_act_ls = execute_bt(tmp_bt, goal, c,
                                                                                verbose=False)
                # if error:
                #     cur_cost = 999999999999999999

                cost_every_exp += cur_cost
                cost_act_num_every_exp += act_num
                self.simu_cost_ls.append(cost_every_exp)
                self.simu_cost_act_num_ls.append(cost_act_num_every_exp)
                if cost_act_num_every_exp!=0:
                    self.cost_act_num_ratio.append(cost_every_exp/cost_act_num_every_exp)
                else:
                    self.cost_act_num_ratio.append(0)
                if len(self.expanded)>self.max_expanded_num:
                    bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                              cond_to_condActSeq)
                    return bt, min_cost, self.time_limit_exceeded





            # ====================== Action Trasvers ============================ #
            traversed_current = []
            # Traverse actions to find applicable ones
            self.premise_pairs = self.get_premise_pairs()

            for premise_pair in self.premise_pairs:
                act = premise_pair.act_leaf.content
                c_attr = premise_pair.cond_leaf.content

                self.premise_pair = premise_pair
                premise_cost = self.calculate_cost()



                c_attr_node = Leaf(type='cond', content=c_attr, min_cost=premise_cost)
                a_attr_node = Leaf(type='act', content=act,
                                   min_cost=premise_cost)
                new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)

                self.put_pair(new_pair)

                # self.nodes.append(new_pair)

                # Directly expand these actions to the behavior tree
                # Build the sequence structure of actions
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child([c_attr_node, a_attr_node])
                # Add the sequence structure to the subtree
                subtree.add_child([sequence_structure])
                
                self.tree_size_ls.append(c_attr)

                if self.output_just_best:
                    cond_to_condActSeq[new_pair] = sequence_structure
                    child_to_parent[new_pair] = current_pair

                
                self.traversed_state_num += 1
                self.traversed_act.append(act.name)
                # Put all action nodes that meet the conditions into the list
                traversed_current.append(c_attr)
                
                
                # Break out here
                if c_attr <= start and not self.is_robust_expand:

                    if self.continue_expand:
                        if not self.is_robust_expand:
                            self.is_robust_expand = True
                            if self.use_robust:
                                self.create_robust_literal_value_dict()
                    else:
                        parent_of_c = current_pair.cond_leaf.parent
                        parent_of_c.children[0] = subtree
                        bt = self.post_processing(new_pair, goal_cond_act_pair, subtree, bt,
                                                  child_to_parent, cond_to_condActSeq)
                        if self.exp:
                            self.expanded_percentages.append(
                                calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                            self.traversed_percentages.append(
                                calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))
                        return bt, current_mincost + act.cost,self.time_limit_exceeded

                self.traversed_state_num += 1

                if self.verbose:
                    print("———— -- Action={} meets conditions, new condition={}".format(act.name, c_attr))

            # Time out
            if (self.time_limit != None and time.time() - start_time > self.time_limit) or (self.max_expanded_num is not None and len(self.expanded)>=self.max_expanded_num):
                self.time_limit_exceeded = True
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq,success=False)
                return bt, min_cost, self.time_limit_exceeded


            # Replace the original condition node c_node with the expanded subtree
            parent_of_c = current_pair.cond_leaf.parent
            parent_of_c.children[0] = subtree
            self.traversed.append(c)
            # ====================== End Action Trasvers ============================ #

        # self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                  cond_to_condActSeq, success=False)
        self.bt_without_merge = bt

        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=merge_time)

        if self.verbose:
            print("Error: Couldn't find successful bt!")
            print("Algorithm ends!\n")

        return bt, min_cost,self.time_limit_exceeded




