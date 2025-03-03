import gym
from stable_baselines3 import PPO
from GridBasedPathPlanning.RL.RL_ARCHIVE.GridMapGymEnv import GridMapGymEnv


EPISODES = 100
TIMESTEPS = 10000
MODEL_NAME = "MODEL_20240124_164536/770000"
MODEL_PATH = f"RL/MODEL/{MODEL_NAME}.zip"

env_params = {'GridSize': 100}


env = GridMapGymEnv(params=env_params, render_mode='human')    # Create the environment
env.reset()
model = PPO.load(MODEL_PATH,env=env)

##### TRAINING LOOP #####
for epsiode in range(EPISODES):
    obs = env.reset()
    done = False

    while not done:
        env.render()

        if type(obs) is tuple: obs = obs[0]     # For some unknown reason, in the first iteration the returned type of obs is a tuple, then it is a dict ????

        action, _state = model.predict(observation=obs,deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
    
    print(f"Number of collisions: {info['collision_num']}")

env.close()


