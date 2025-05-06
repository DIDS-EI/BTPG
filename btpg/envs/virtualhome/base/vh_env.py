import subprocess
import time

from btpg.envs.base.env import Env
from btpg.envs.virtualhome.simulation.unity_simulator import UnityCommunication
from btpg.utils import ROOT_PATH

import atexit

class VHEnv(Env):
    agent_num = 1

    headless = True

    # launch simulator
    simulator_path = f'{ROOT_PATH}/../simulators/virtualhome/windows/VirtualHome.exe'

    behavior_lib_path = f"{ROOT_PATH}/envs/virtualhome/exec_lib"


    def __init__(self):
        if not self.headless:
            self.launch_simulator()
        super().__init__()

    def reset(self):
        raise NotImplementedError

    def task_finished(self):
        raise NotImplementedError


    # def launch_simulator(self):
    #     self.comm = UnityCommunication()
    #     self.simulator_process = subprocess.Popen(self.simulator_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,start_new_session=True)
    #
    #     atexit.register(self.close)

    def launch_simulator(self):
        # 设置全屏分辨率为 2560x160
        simulator_command = [self.simulator_path, '-screen-fullscreen', '1', '-screen-width', '2560', '-screen-height', '1600']

        self.comm = UnityCommunication()
        self.simulator_process = subprocess.Popen(simulator_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

        atexit.register(self.close)

    def load_scenario(self,scenario_id):
        simulator_launched = False
        while not simulator_launched:
            try:
                self.comm.reset(scenario_id)
                # self.comm.activate_physics()
                simulator_launched = True
            except:
                pass

    def run_script(self,script,verbose=False,camera_mode="PERSON_FROM_BACK"):
        # camera_mode = "FIRST_PERSON"#"PERSON_FROM_BACK" #"FIRST_PERSON" #"AUTO"
        success, message = self.comm.render_script(script, recording=True,skip_animation=False, frame_rate=10, camera_mode=[camera_mode],
                               find_solution=True)

        # Check whether the command was executed successfully
        if verbose:
            if success:
                print(f"'Successfully.")
            else:
                print(f"'Failed,{message}'.")

    def close(self):
        time.sleep(1)
        self.simulator_process.terminate()

