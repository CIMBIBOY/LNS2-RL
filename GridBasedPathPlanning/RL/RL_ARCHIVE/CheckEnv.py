import os, sys
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if module_path not in sys.path: sys.path.append(module_path)

from GridBasedPathPlanning.RL.RL_ARCHIVE.GridMapGymEnv import GridMapGymEnv



# CHECK 1
env_params = {'GridSize': 100}
env = GridMapGymEnv(params=env_params)
# It will check your custom environment and output additional warnings if needed
check_env(env)




# CHECK 2
env_params = {'GridSize': 100,}
env = GridMapGymEnv(params=env_params, render_mode='human')
env.reset()

model = PPO('MultiInputPolicy', env, verbose=1)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)