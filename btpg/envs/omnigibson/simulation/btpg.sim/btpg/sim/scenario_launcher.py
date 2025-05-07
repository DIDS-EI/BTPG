from .utils import SharedStatus, get_isaacsim_asset
import os
from .scenarios.base import BaseScenario
from .scenarios.omnigibson_tiago import OmnigibsonTiago
from .scenarios.omnigibson_base import OmnigibsonBase

class ScenarioLauncher:
    def __init__(self,shared_status:SharedStatus):
        self.scenario:BaseScenario = OmnigibsonTiago(shared_status)

    def reset(self):
        self.scenario.reset()

    def load_example_assets(self):
        return self.scenario.load_example_assets()

    def setup(self):
        self.scenario.setup()

    def step(self, action):
        return self.scenario.step(action)

    def set_world(self, world):
        self.scenario.set_world(world)
    
    def close(self):
        self.scenario.close()