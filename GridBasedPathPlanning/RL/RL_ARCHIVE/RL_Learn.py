import os
import numpy as np
import gym
import torch
import time
import shutil
from datetime import datetime
from stable_baselines3 import PPO
from GridBasedPathPlanning.RL.RL_ARCHIVE.GridMapGymEnv import GridMapGymEnv


if __name__ == '__main__':

    if torch.cuda.is_available():
        print("GPU is available")

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    EPISODES = 100
    TIMESTEPS = 10000
    MODEL_DIR = f"RL/MODEL/MODEL_{date_time}"
    LOG_DIR = "RL/LOG"

    os.makedirs(LOG_DIR,exist_ok=True)
    os.makedirs(MODEL_DIR,exist_ok=True)
    shutil.copy("Environment/GridMapEnv.py",f"{MODEL_DIR}/GridMapEnv.py")
    shutil.copy("Environment/GridMapGymEnv.py",f"{MODEL_DIR}/GridmapGymEnv.py")
    shutil.copy("Environment/Object.py",f"{MODEL_DIR}/Object.py")
    shutil.copy("RL_Learn.py",f"{MODEL_DIR}/RL_Learn.py")
    shutil.copy("RL_Load.py",f"{MODEL_DIR}/RL_Load.py")


    env_params = {'GridSize': 100}

    #from stable_baselines3.common.env_util import make_vec_env
    #env = make_vec_env(GridMapEnv, env_kwargs={"params":env_params}, n_envs=16, seed=0)   # Create the vectorized environment

    env = GridMapGymEnv(params=env_params)      # Create the environment
    env.reset()
    model = PPO("MultiInputPolicy",          # Create the RL agent
                env,
                verbose=0,
                tensorboard_log=LOG_DIR,
                device='cpu')

    ##### TRAINING LOOP #####
    for epsiode in range(EPISODES):
        model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name=f"LOG_{date_time}")
        model.save(f"{MODEL_DIR}/{TIMESTEPS*epsiode}")

    env.close()
