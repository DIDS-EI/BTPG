import time

import btgym
from btgym.algos.llm_client.tools import goal_transfer_str
from btgym.algos.bt_planning.main_interface import BTExpInterface
from btgym.utils.tools import *
from btgym.utils.goal_generator.vh_gen import VirtualHomeGoalGen


max_goal_num=5
diffcult_type= "single" #"single"  #"mix" "multi"
scene = "VH" # RH RHS RW

# ===================== VirtualHome ========================
goal_gen = VirtualHomeGoalGen()
goal_ls = goal_gen.random_generate_goals(max_goal_num ,diffcult_type=diffcult_type)
# for goal in goal_ls:
#     print(goal)

env, cur_cond_set = setup_environment(scene)


# for i,goal_str in enumerate(goal_ls):
for i,goal_str in enumerate(['IsIn_milk_fridge']):
    print("i:", i, "goal_str:", goal_str)

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=[], key_predicates=[],
                          key_objects=[],
                          selected_algorithm="opt", mode="big",
                          act_tree_verbose=False, time_limit=15,
                          heuristic_choice=0,output_just_best=True)

    goal_set = goal_transfer_str(goal_str)

    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()
    planning_time_total = end_time - start_time

    time_limit_exceeded = algo.algo.time_limit_exceeded

    ptml_string, cost, expanded_num = algo.post_process()
    error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

    print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
          "\x1b[31mERROR\x1b[0m" if error else "",
          "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
    print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

# visualization
file_name = "tree"
file_path = f'./{file_name}.btml'
with open(file_path, 'w') as file:
    file.write(ptml_string)
# read and execute
from btgym import BehaviorTree
bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
# bt.print()
bt.draw()