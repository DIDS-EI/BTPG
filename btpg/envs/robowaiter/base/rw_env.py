import subprocess
import time
import psutil
from btpg.envs.base.env import Env
from btpg.utils import ROOT_PATH

import atexit
from btpg.envs.robowaiter.scene import Scene
import dataclasses

@dataclasses.dataclass
class HeadlessScene:
    headless = True

class RWEnv(Env):
    agent_num = 1

    # launch simulator
    simulator_path = f'{ROOT_PATH}/../simulators/robowaiter/CafeSimulator/CafeSimulator.exe'

    behavior_lib_path = f"{ROOT_PATH}/envs/robowaiter/exec_lib"


    def __init__(self):
        if not self.headless:
            self.launch_simulator()
            self.scene = Scene()
        else:
            self.scene = HeadlessScene()
        super().__init__()

    def reset(self):
        raise NotImplementedError

    def task_finished(self):
        raise NotImplementedError


    def launch_simulator(self):
        print('Launching simulator...')
        self.simulator_process = subprocess.Popen(self.simulator_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,start_new_session=True)
        atexit.register(self.close)
        # Check for startup flag
        while True:
            line = self.simulator_process.stdout.readline().decode()
            # print(line)
            if "Engine is initialized" in line:
                break
        print("Simulator is ready.")

    def load_scenario(self,scenario_id):
        simulator_launched = False
        while not simulator_launched:
            try:
                # self.comm.reset(scenario_id)
                # self.comm.activate_physics()
                self.scene.reset()
                simulator_launched = True
            except:
                pass

    def close(self):
        time.sleep(1)
        # self.simulator_process.terminate()
        # parent_pid = self.simulator_process.pid
        # parent = psutil.Process(parent_pid)
        # for child in parent.children(recursive=True):
        #     child.kill()
        # parent.kill()

        parent_pid = self.simulator_process.pid  # Get the parent process ID
        if psutil.pid_exists(parent_pid):  # Check if the parent process still exists
            try:
                parent = psutil.Process(parent_pid)
                # Attempt to terminate all child processes
                for child in parent.children(recursive=True):
                    if psutil.pid_exists(child.pid):  # Check if the child process exists
                        child.kill()
                # After handling all child processes, attempt to terminate the parent process
                if psutil.pid_exists(parent_pid):  # Check again if the parent process still exists
                    parent.kill()
            except psutil.NoSuchProcess:
                print(f"Process with PID {parent_pid} no longer exists.")
            except Exception as e:
                print(f"An error occurred while trying to terminate the simulator: {e}")
