import copy
import heapq
import numpy as np
import random
from btpg.algos.base.tools import *
from btpg.algos.base.btp_base_test import BTPlannerBaseTest,CondActPair
from collections import deque
from btpg.utils.string_format import parse_predicate_logic
seed = 0
random.seed(seed)
np.random.seed(seed)


class UHBTPTest(BTPlannerBaseTest):
    def __init__(self, **kwargs):
        # Ensure bt_merge is explicitly set to False before calling superclass constructor
        super().__init__(bt_merge=False,**kwargs)
        self.max_heuristic_level = None
        info_dict = kwargs['info_dict']
        arg_set = info_dict['arg_set']
        self.theory_priority_act_ls = kwargs.get('theory_priority_act_ls',None)

        self.priority_cond_set = info_dict.get('priority_cond_set',set())
        self.priority_act_ls = info_dict.get('priority_act_ls',[])

        self.use_delete_relaxation =  'delete' in arg_set
        self.use_solution =  'solution' in arg_set
        self.use_llm =  'llm' in arg_set
        self.use_rllm =  'rllm' in arg_set
        self.use_llm_additive =  'llm_additive' in arg_set
        self.use_value =  'value' in arg_set
        self.use_robust =  'robust' in arg_set

    def put_pair(self,pair):
        heapq.heappush(self.nodes, pair)

    def pop_pair(self):
        current_pair = heapq.heappop(self.nodes)
        self.current_pair = current_pair
        return current_pair

    def create_literal_list(self):
        literal_set = self.goal | self.start
        for a in self.actions:
            literal_set |= a.pre | a.add | a.del_set
        self.literal_list = sorted(list(literal_set))

    def create_heuristic_list(self):
        self.heuristic_list = [copy.deepcopy(self.start)]

        current_condition = copy.deepcopy(self.start)

        while not self.goal <= current_condition:
            next_condition = copy.deepcopy(current_condition)
            for act in self.actions:
                if act.pre <= current_condition:
                    next_condition |= act.add

            self.heuristic_list.append(next_condition)
            current_condition = next_condition

        self.max_heuristic_level = len(self.heuristic_list)


    def create_robust_heuristic_list(self):
        self.robust_heuristic_list = [copy.deepcopy(self.start)]

        current_condition = copy.deepcopy(self.goal)

        last_condition = set()
        while last_condition != current_condition:
            last_condition = copy.deepcopy(current_condition)
            next_condition = copy.deepcopy(current_condition)
            for act in self.actions:
                if act.add <= current_condition:
                    next_condition |= act.pre

            self.robust_heuristic_list.append(next_condition)
            current_condition = next_condition

        self.max_robust_heuristic_level = len(self.robust_heuristic_list)

    def refresh_node_queue(self):

        # print(sorted([pair.cond_leaf.min_cost for pair in self.nodes]))
        # old_nodes = copy.deepcopy(self.nodes)
        # self.nodes = []
        for pair in self.nodes:
            # self.premise_pair = pair
            pair.cond_leaf.min_cost = self.calculate_cost()
            # self.put_pair(pair)
        # print(sorted([pair.cond_leaf.min_cost for pair in self.nodes]))


    def create_robust_literal_value_dict(self):
        self.create_robust_heuristic_list()

        # self.literal_value_dict = {}
        # self.literal_list += list(self.priority_cond_set)

        # for literal in self.literal_list:
        #     predicate,args = parse_predicate_logic(literal)
        #     self.literal_value_dict[literal] =  len(set(args) - self.relevant_objects)
        #     if self.use_llm_additive:
        #         if literal in self.llm_literal_set:
        #             self.literal_value_dict[literal] -= 1

        # delete relaxation
        level_condition_list = [self.robust_heuristic_list[0]]
        for level in range(1,self.max_robust_heuristic_level):
            level_condition_list.append(self.robust_heuristic_list[level] - self.robust_heuristic_list[level-1])

        for level, level_literal in enumerate(level_condition_list):
            for literal in level_literal:
                self.literal_value_dict[literal] = max(1,self.literal_value_dict[literal]-level-1)

        # for literal in self.start:
        #     self.literal_value_dict[literal] = 1

        if self.use_rllm:
            if len(self.priority_cond_set)>0:
                for literal in self.priority_cond_set:
                    if literal in self.literal_value_dict:
                        self.literal_value_dict[literal] = 1
                    else:
                        self.literal_value_dict[literal] = 1



        # for literal in self.priority_cond_set:
        #     self.literal_value_dict[literal] = 0
        #
        #
        # if self.use_llm:
        #     for literal in self.llm_literal_set:
        #         self.literal_value_dict[literal] = 0
        self.refresh_node_queue()


    def create_literal_value_dict(self):
        self.create_heuristic_list()

        self.literal_value_dict = {}

        relevant_objects = set()
        relevant_objects = {"self"}
        for goal_literal in self.goal:
            predicate, args = parse_predicate_logic(goal_literal)
            relevant_objects.update(args)
        self.relevant_objects = relevant_objects

        for literal in self.literal_list:
            self.literal_value_dict[literal] = self.max_heuristic_level

        if self.use_value:
            for literal in self.literal_list:
                predicate,args = parse_predicate_logic(literal)
                self.literal_value_dict[literal] =  len(set(args) - relevant_objects)
                # if self.use_llm:
                #     if literal in self.llm_literal_set:
                #         self.literal_value_dict[literal] -= 1



        level_condition_list = [self.heuristic_list[0]]
        for level in range(1,self.max_heuristic_level):
            level_condition_list.append(self.heuristic_list[level] - self.heuristic_list[level-1])

        # level_condition_list.reverse()
        if self.use_delete_relaxation:
            for level, level_literal in enumerate(level_condition_list):
                for literal in level_literal:
                    self.literal_value_dict[literal] += level+1

        # if self.use_llm:
        #     for i in range(len(self.priority_act_ls)):
        #         a_name = self.priority_act_ls[i]
        #         a = self.llm_literal_dict[a_name]
        #         for literal in a.pre:
        #             self.literal_value_dict[literal] -= 1+ (i+1) / len(self.priority_act_ls)

        if self.use_llm:
            for literal in self.llm_literal_set:
                self.literal_value_dict[literal] = max(0,self.literal_value_dict[literal]-1)


        # for level, condition in enumerate(self.heuristic_list):
        #
        #     for literal in condition

    def create_llm_literal_list(self):
        self.llm_literal_set = set()
        self.llm_literal_dict = {}
        for a in self.actions:
            if a.name in self.priority_act_ls:
                self.llm_literal_set |= a.pre
                self.llm_literal_dict[a.name] = a


    def pre_process(self):
        if self.use_solution:
            self.calculate_cost = self.heuristic
        else:
            self.calculate_cost = self.sum_cost

        self.create_literal_list()
        self.create_llm_literal_list()

        if self.use_solution:
            self.create_literal_value_dict()

        # if self.use_robust:
        #
        # else:



    def heuristic(self):
        premise_condition = self.premise_pair.cond_leaf.content

        value = 0
        for literal in premise_condition:
            value += self.literal_value_dict.get(literal,self.max_heuristic_level)
        return value



    # def delete_relaxation_all(self):
    #     premise_condition = self.premise_pair.cond_leaf.content
    #
    #     for level in range(self.max_heuristic_level):
    #         if premise_condition <= self.heuristic_list[level]:
    #             return self.max_heuristic_level - level
    #
    #     return self.max_heuristic_level




    def post_processing(self,pair_node,g_cond_anc_pair,subtree,bt,child_to_parent,cond_to_condActSeq,success=True):
        '''
        Process the summary work after the algorithm ends.
        '''
        if self.output_just_best:

            if success:
                new_bt = ControlBT(type='cond')
                goal_condition_node = Leaf(type='cond', content=g_cond_anc_pair.cond_leaf.content, min_cost=0)
                # Retain the expanded nodes in the subtree first
                new_subtree = ControlBT(type='?')
                new_subtree.add_child([copy.deepcopy(goal_condition_node)])

                # =================================
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
                    new_subtree.add_child([copy.deepcopy(tmp_seq_struct)])

                # If it is not an empty tree
                new_bt.add_child([new_subtree])
                bt = copy.deepcopy(new_bt)
            else:
                new_bt = ControlBT(type='cond')
                new_subtree = ControlBT(type='?')
                goal_condition_node = Leaf(type='cond', content=g_cond_anc_pair.cond_leaf.content, min_cost=0)
                new_subtree.add_child([copy.deepcopy(goal_condition_node)])
                new_bt.add_child([new_subtree])
                bt = copy.deepcopy(new_bt)


        # self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt

        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=self.merge_time)
        return bt
