import pysocialforce as psf

import logging
import os

# 1. Force the environment variable to off inside the script
os.environ['NUMBA_DEBUG'] = "0"

# 2. Specifically tell the numba logger to ignore anything below WARNING
logging.getLogger('numba').setLevel(logging.WARNING)

class SFM:
    def __init__(self, initial_state, obs=None, groups=None):
        self.name = 'SFM'
        self.sim = psf.Simulator(
            initial_state,
            groups=groups,
            obstacles=obs,
        )

    def update(self, state):
        self.sim.update_state(state)

    def predict(self):
        desired_vel = self.sim.step_once_desired_velocity()
        return desired_vel[:-1, :]   # remove the last vel because it belongs to the ego-agent(robot)