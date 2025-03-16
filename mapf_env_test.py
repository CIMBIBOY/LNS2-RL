import os
import random
from PIL import Image
import numpy as np
from mapf_gym import CL_MAPFEnv  # adjust import as needed

def test_mapf_env(num_steps=20, output_dir="test_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and reset the environment.
    env = CL_MAPFEnv(env_id=1)
    env._global_reset(cl_num_task=0)
    print("Global reset complete.")
    
    # Call local reset: for testing, use all agents (or choose a subset)
    env._local_reset(local_num_agents=7, first_time=True)
    print("Local reset complete.")

    # Optionally, if your world doesn't initialize local_agents_poss, do this:
    if not hasattr(env.world, "local_agents_poss"):
        env.world.local_agents = list(range(7))
        env.world.local_agents_poss = [env.world.agents_poss[i] for i in range(7)]
        print("Initialized local_agents and local_agents_poss for testing.")
    
    # Save an initial RGB image.
    env.save_rgb_image(os.path.join(output_dir, "step_000.png"))
    
    # Iterate and step the environment.
    for step in range(1, num_steps):
        actions = []
        for agent in range(env.local_num_agents):
            try:
                valid = env.list_next_valid_actions(agent)
            except Exception as e:
                print(f"Error getting valid actions for agent {agent} at step {step}: {e}")
                valid = [4]  # default to idle if error occurs
            if valid:
                actions.append(random.choice(valid))
            else:
                actions.append(4)
                
        state, vector, rewards, done, next_valid_actions, num_on_goal, num_dynamic_collide, num_agent_collide, success, real_r = env.joint_step(actions)
        
        out_file = os.path.join(output_dir, f"step_{step:03d}.png")
        env.save_rgb_image(out_file)
        print(f"Saved image for step {step} to {out_file}")
        print("Global num agents:", env.global_num_agent)
        print("Number of goals:", len(env.goal_list))
        
        if done:
            print("Episode finished early.")
            break

if __name__ == "__main__":
    test_mapf_env(num_steps=20, output_dir="test_images")