from btpg.envs.robowaiter.exec_lib._base.rw_action import RWAction
import btpg
from btpg.utils.tools import collect_action_nodes



# ===================== RoboWaiter ========================
# from btpg.envs.robowaiter.exec_lib._base.rw_action import RWAction
# env = btpg.make("RW")
# cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)', 'Holding(Nothing)'}
# cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
# cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
# big_actions = collect_action_nodes(env.behavior_lib)
# print(f"Collected a total of {len(RWAction.AllObject)} objects")


# ===================== VirtualHome ========================
# from btpg.envs.virtualhome.exec_lib._base.vh_action import VHAction

# env = btpg.make("VH")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
#                                               "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
# big_actions = collect_action_nodes(env.behavior_lib)
# print(f"Collected a total of {len(VHAction.AllObject)} objects")


# ===================== Omnigibson ========================
# from btpg.envs.omnigibson.exec_lib._base.og_action import OGAction
# env = btpg.make("OG")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
#                                               "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in OGAction.CAN_OPEN}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in OGAction.HAS_PLUG}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in OGAction.HAS_SWITCH}
# big_actions = collect_action_nodes(env.behavior_lib)


# ===================== RobotHow ========================
from btpg.envs.robothow.exec_lib._base.rh_action import RHAction as RH

env = btpg.make("RH")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                              "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in RH.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RH.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in RH.HAS_PLUG}
print(f"Collected a total of {len(RH.AllObject)} objects")
big_actions = collect_action_nodes(env.behavior_lib)


# Calculate the number of finite states
conds_set=set()
for act in big_actions:
    conds_set |= act.pre
    conds_set |= act.del_set
    conds_set |= act.add
print("len(conds_set)", len(conds_set))
