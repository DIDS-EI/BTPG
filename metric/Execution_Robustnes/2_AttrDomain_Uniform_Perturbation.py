import copy
import os
import random
from btpg.utils import ROOT_PATH
os.chdir(f'{ROOT_PATH}/../test_exp')
from tools import modify_condition_set_Random_Perturbations
from btpg.utils.tools import setup_environment
import time
import re
import pandas as pd
import btpg
from btpg.utils.tools import collect_action_nodes
from btpg.utils.read_dataset import read_dataset
from btpg.algos.llm_client.tools import goal_transfer_str
from btpg.algos.bt_planning.bt_planner_interface import BTPlannerInterface
from btpg.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
import concurrent.futures
from btpg.envs.robowaiter.exec_lib._base.rw_action import RWAction
from btpg.envs.virtualhome.exec_lib._base.vh_action import VHAction
from btpg.envs.omnigibson.exec_lib._base.og_action import OGAction
from btpg.envs.robothow.exec_lib._base.rh_action import RHAction

SENCE_ACT_DIC={"RW":RWAction,
               "VH":VHAction,
               "RHS":OGAction,
               "RH":RHAction}

def get_SR(scene, algo_str, just_best,exe_times=5,data_num=100,p=0.2,difficulty="multi"):

    AVG_SR = 0

    # 导入数据
    data_path = f"{ROOT_PATH}/../test_exp/data/{scene}_{difficulty}_100_processed_data.txt"
    data = read_dataset(data_path)
    llm_data_path = f"{ROOT_PATH}/../test_exp/llm_data/{scene}_{difficulty}_100_llm_data.txt"
    llm_data = read_dataset(llm_data_path)
    env, cur_cond_set = setup_environment(scene)

    algo_str_complete = algo_str
    heuristic_choice = -1
    if algo_str == "hobtea_h0" or algo_str == "hobtea_h0_llm":
        heuristic_choice = 0
    elif algo_str == "hobtea_h1":
        heuristic_choice = 1
    if algo_str in ['hobtea_h0', 'hobtea_h1', "hobtea_h0_llm"]: algo_str = 'hobtea'


    for i, (d, ld) in enumerate(zip(data[:data_num], llm_data[:data_num])):
        print("data:", i, "scene:",scene, "algo:",algo_str_complete, "just_best:", just_best)
        goal_str = ' & '.join(d["Goals"])
        goal_set = goal_transfer_str(goal_str)
        opt_act = act_str_process(d['Optimal Actions'], already_split=True)

        priority_opt_act = []
        if algo_str_complete == "hobtea_h0_llm":
            priority_opt_act = act_str_process(ld['Optimal Actions'], already_split=True)
            # print("llm_opt_act:", priority_opt_act)
            # print("opt_act:", opt_act)
        elif "hobtea" in algo_str_complete:
            priority_opt_act = opt_act

        algo = BTPlannerInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                              priority_act_ls=priority_opt_act, key_predicates=[],
                              key_objects=[],
                              selected_algorithm=algo_str, mode="big",
                              act_tree_verbose=False, time_limit=5,
                              heuristic_choice=heuristic_choice, exp_record=False, output_just_best=just_best,
                              theory_priority_act_ls=opt_act)

        goal_set = goal_transfer_str(goal_str)
        start_time = time.time()
        algo.process(goal_set)
        end_time = time.time()
        planning_time_total = end_time - start_time
        time_limit_exceeded = algo.algo.time_limit_exceeded
        # ptml_string, cost, expanded_num = algo.post_process(ptml_string=False)
        # error, state, act_num, current_cost, record_act_ls, ticks = algo.execute_bt(goal_set[0], cur_cond_set,
        #                                                                             verbose=False)
        # print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
        #       "\x1b[31mERROR\x1b[0m" if error else "",
        #       "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
        # print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total,
        #       "ticks:", ticks)

        # 跑算法
        # 提取出obj
        objects = []
        pattern = re.compile(r'\((.*?)\)')
        for expr in goal_set[0]:
            match = pattern.search(expr)
            if match:
                objects.extend(match.group(1).split(','))
        successful_executions = 0  # Used to track the number of successful (non-error) executions
        # Randomly generate exe_times initial states to see which one can reach the goal
        for i in range(exe_times):
            # new_cur_state = modify_condition_set(scene,SENCE_ACT_DIC[scene], cur_cond_set,objects)
            # error, state, act_num, current_cost, record_act_ls, ticks = algo.execute_bt(goal_set[0], new_cur_state,
            #                                                                             verbose=False)

            # new_cur_state = modify_condition_set(scene,SENCE_ACT_DIC[scene], cur_cond_set,objects)

            new_cur_state = modify_condition_set_Random_Perturbations(scene, SENCE_ACT_DIC[scene], cur_cond_set, objects, p=p)
            error, state, act_num, current_cost, record_act_ls, ticks = algo.execute_bt_Random_Perturbations(scene, SENCE_ACT_DIC[scene],objects,\
                                                                                                             goal_set[0], new_cur_state,
                                                                                        verbose=False, p=p)


            # Check for errors, and if none, increase the success count
            if not error:
                successful_executions += 1
        # Calculate the proportion of non-error executions
        success_ratio = successful_executions / exe_times
        AVG_SR += success_ratio

    AVG_SR = AVG_SR / data_num
    print("Proportion of successful executions (non-error): {:.2%}".format(AVG_SR))
    return round(AVG_SR, 2)


def run_simulation(scene, algo_str, just_best):
    return scene, algo_str, get_SR(scene, algo_str, just_best, exe_times=5, data_num=data_num, p=p, difficulty=difficulty)



algorithms = ['hobtea_h0', 'hobtea_h0_llm', 'obtea', 'bfs']  # 'hobtea_h0', 'hobtea_h1', 'obtea', 'bfs', 'dfs'
scenes = ['RW', 'VH' , 'RHS' ,'RH']  # 'RH', 'RHS', 'RW', 'VH'
just_best_bts = [False] # True, False


data_num=100
p=0.5
difficulty = "multi"


index = [f'{algo_str}_{tb}' for tb in ['T', 'F'] for algo_str in algorithms ]
df = pd.DataFrame(index=index, columns=scenes)
# for just_best in just_best_bts:
#     for algo_str in algorithms:
#         index_key = f'{algo_str}_{"T" if just_best else "F"}'
#         for scene in scenes:
#             df.at[index_key, scene] = get_SR(scene, algo_str, just_best,exe_times=5,data_num=data_num,p=p,difficulty=difficulty)
with concurrent.futures.ThreadPoolExecutor() as executor: #max_workers=4
    future_to_scene_algo = {}
    for just_best in just_best_bts:
        for algo_str in algorithms:
            for scene in scenes:
                index_key = f'{algo_str}_{"T" if just_best else "F"}'
                future = executor.submit(run_simulation, scene, algo_str, just_best)
                future_to_scene_algo[future] = (index_key, scene)

    # Retrieve results and update DataFrame
    for future in concurrent.futures.as_completed(future_to_scene_algo):
        index_key, scene = future_to_scene_algo[future]
        _, _, sr = future.result()
        df.at[index_key, scene] = sr


formatted_string = df.to_csv(sep='\t')
print(formatted_string)
print("----------------------")
print(df)

# Save the DataFrame to a CSV file
csv_file_path = f"{ROOT_PATH}/../test_exp/Execution_Robustnes/2_changes_all_p={p}_t=5_d={data_num}_{difficulty}.csv"  # Define your CSV file path
df.to_csv(csv_file_path)  # Save as a TSV (Tab-separated values) file
print(csv_file_path)