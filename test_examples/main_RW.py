import time

import btpg
from btpg.algos.llm_client.tools import goal_transfer_str
from btpg.algos.bt_planning.bt_planner_interface import BTPlannerInterface
from btpg.utils.tools import *
from btpg.utils.goal_generator.rw_gen import RoboWaiterGoalGen
import os
DIR = os.path.dirname(os.path.abspath(__file__))


max_goal_num=5
diffcult_type= "mix" #"single"  #"mix" "multi"
scene = "RW" # RH RHS RW

headless = False


# ===================== RoboWaiter ========================
goal_gen = RoboWaiterGoalGen()
goal_ls = goal_gen.random_generate_goals(max_goal_num ,diffcult_type=diffcult_type)
for goal in goal_ls:
    print(goal)

env, cur_cond_set = setup_environment(scene)

# goal_str ='On_Coffee_Table3 & Active_AC'
# goal_str ='RobotNear_WaterStation'
# goal_str ='IsClean_Floor'
# 'On_Water_WaterStation'
goal_str = 'Low_ACTemperature'
print("goal_str:", goal_str)

algo = BTPlannerInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                      priority_act_ls=[], key_predicates=[],
                      key_objects=[],
                      selected_algorithm="hbtp", mode="big",
                      time_limit=15,
                      heuristic_choice=0,output_just_best=False)

goal_set = goal_transfer_str(goal_str)

start_time = time.time()
algo.process(goal_set)
end_time = time.time()
planning_time_total = end_time - start_time

time_limit_exceeded = algo.algo.time_limit_exceeded

btml_string, cost, expanded_num = algo.post_process()
error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
      "\x1b[31mERROR\x1b[0m" if error else "",
      "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)


# visualization
file_name = "tree"
file_path = f'{DIR}/{file_name}.btml'
print(f"保存文件到路径: {file_path}")
with open(file_path, 'w') as file:
    file.write(btml_string)
# read and execute
from btpg import BehaviorTree
print(f"尝试加载文件: {file_path}")
bt = BehaviorTree(file_path, env.behavior_lib)
bt.print()
bt.draw()

# Simulate execution in a simulated scenario.
# goal_str = 'IsIn_milk_fridge & IsClose_fridge'
# goal_str = 'IsOn_bananas_kitchentable'
goal = goal_transfer_str(goal_str)[0]
print(f"goal: {goal}") # {'IsIn(milk,fridge)', 'IsClose(fridge)'}




if scene in ["VH","RW"]:
    if not headless:
        env.agents[0].bind_bt(bt)
        env.reset()
        time.sleep(5)
        is_finished = False
        while not is_finished:
            is_finished = env.step()
            if goal <= env.agents[0].condition_set:
                is_finished=True
        env.close()
else:
    error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=True)
