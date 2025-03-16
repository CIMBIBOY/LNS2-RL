from PIL import Image
import numpy as np
from GridBasedPathPlanning.RL_Environment.environment.dynamic_obstacle import update_agent
from GridBasedPathPlanning.RL_Environment.environment.map_generator import start_end_points, global_guidance, map_to_value
from GridBasedPathPlanning.Plan.TJPS import TJPS
from GridBasedPathPlanning.RL.helpers.utils import symmetric_pad_array, save_array_as_png
from GridBasedPathPlanning.RL_Environment.environment.ViNT_helpers import save_numpy_to_txt, binary_img_gray_white, binary_image_match_green, binary_image_match_black, save_grid_with_name
import os
import datetime
import pickle
import math
import imageio
import pygame
import time
from collections import deque
import torch
from gym.spaces import Box, Discrete
from gym.utils import seeding
import random
from GridBasedPathPlanning.Environment.GridMapEnv import GridMapEnv, GridToRGB
import matplotlib.colors as mcolors


def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

def hue_to_rgb(hue_array):
    # Assume full saturation (1.0) and full value (1.0)
    saturation = np.ones_like(hue_array)
    value = np.ones_like(hue_array)

    # Stack hue, saturation, and value into an HSV image
    hsv_image = np.stack([hue_array, saturation, value], axis=-1)

    # Convert HSV to RGB using matplotlib's hsv_to_rgb function
    rgb_image = mcolors.hsv_to_rgb(hsv_image)

    return rgb_image

class WarehouseEnvironment:

    def __init__(self, env_idx, height = 48, width = 48, amr_count = 1, max_amr = 10, agent_idx = 0, local_fov = 15, time_dimension = 1, pygame_render = True, seed = None, FRAMES = 200, curriculum = 100, static_map = "GridBasedPathPlanning/RL_Environment/data/cleaned_empty/empty-48-48-random-10_60_agents.png"):

        # Initial map address
        self.idx = env_idx
        self.name = "g2rl"
        # self.map_path = "data/cleaned_empty/empty-48-48-random-10_60_agents.png"
        self.grid = GridMapEnv(grid_size=(height,width)) # Create the Grid environment
        # Set the grid with static map and initialize GridSequence
        self.set_grid(static_map, 0, amr_count, FRAMES)
        # Dynamic objects start and max number
        self.max_amr = max_amr
        self.amr_count = len(self.grid.objects)
        # map size
        self.height = height
        self.width = width
        # state space dimension
        self.n_states = height * width
        # Number of historical observations to use
        self.Nt = time_dimension
        # Buffer to store past observations to store last 4 observations
        self.observation_history = deque(maxlen=self.Nt)
        # observation space
        self.observation_space = Box(low=0, high=255, shape=(1, self.Nt, local_fov*2, local_fov*2, 4), dtype=np.uint8)
        # action space dim
        self.n_actions = len(self.f_action_space())
        # Define action space
        self.action_space = Discrete(self.n_actions)
        # Agent's path length 
        self.agent_goal = None 
        self.steps = 0
        # Partial field of view size
        self.local_fov = local_fov
        self.time_idx = 0
        self.init_arr = []
        # Array for dynamic objects
        self.agents_coords = []
        self.terminations = np.zeros(4, dtype=int)
        self.last_action = 4
        self.curriculum = curriculum 

        # Agent reached end position count 
        self.episode_count = -1
        self.horizon = 'short'
        self.max_step = 42

        self.data_gen = True
        self.allow_data_gen = True
        self.direction = [0,0]
        self.first_reset = True
        self.map_cap = 64

        self.metadata = {
            'render.modes': ['human'],  # Specifies that 'human' render mode is supported
            'video.frames_per_second': 30  # Frame rate for rendering videos, adjust as needed
        }
        self.pygame_render = pygame_render
        self.screen = None
        self.clock = None

        self.seed(seed)

    def set_grid(self, static_map, map_idx, add_num_dyn_obj, FRAMES):
        self.map_idx = map_idx
        self.map=np.asarray(Image.open(static_map))
        self.map_v = map_to_value(self.map)
        self.grid.SetStaticGrid(self.map_v)
        self.grid.ClearGrid(gridtype="grid_dynamic")
        self.grid.addRandomDynamicObstacles(no_of_obs=add_num_dyn_obj, d_range=(1,6), alpha = 0.4)
        self.frames = FRAMES
        self.grid.ResetDynamicObstacles()
        self.grid_seq = self.grid.GenerateGridSequence(self.frames, reset_after=True)

        self.initial_random_steps = False
        self.was_reached = False
        self.arrived = 0
        self.iter = 0
        self.first_reset = False

        self.info = {
            'R_max_step': False,
            'no_global_guidance': False,
            'goal_reached': False,
            'collision': False,
            'blocked': False,
            'steps': -1,
            # 'path': [],
            'reward': 0,
            'total_arrived': 0
        }


    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)  # Generate a valid seed if none is provided
        elif not (0 <= seed < 2**32):
            raise ValueError("Seed must be between 0 and 2**32 - 1")
        # Ensure seed is an integer
        seed = int(seed)
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]
    
    def reGenerateAndAdd(self):
        self.grid.addRandomDynamicObstacles(no_of_obs=1, d_range=(1,4), alpha = 0.4)
        self.grid_seq = self.grid.GenerateGridSequence(self.frames, reset_after=True)

    def reset(self): 
        # Increment the episode count
        self.episode_count += 1
        self.was_reached = False
        self.data_gen = True
        
        # Reset step count for maximum timesteps
        self.steps = 0
        self.info['steps'] = 0
        
        # Generate new coordinates and paths every 50 episodes
        if self.episode_count == 0 or self.episode_count % self.curriculum == 0:

            self.seed(self.np_random.integers(0, 1000000))
            # Implementing curriculum learning
            if self.arrived >= (self.curriculum * 0.8) and self.amr_count < self.max_amr: 
                self.reGenerateAndAdd()
                self.amr_count += 1
                print(f"Dynamic object added, current count: {self.amr_count}")
            
            # Generate destinations and routes
            self.generate_end_points_and_paths()
            # Initialize map
            self.init_arr = GridToRGB(self.grid.grid)
            if self.init_arr is None or self.init_arr.size == 0:
                raise ValueError("Initialization failed, init_arr is empty or None")

            # save_grid_with_name(self.grid_seq, "GridBasedPathPlanning/RL_Environment/ViNT_data", f"grid_seq_{self.episode_count}.npy")
            # Set dones for new start-goal pair
            self.arrived = 0

            if self.allow_data_gen:
                self.datagen_new()
            if self.first_reset != True:
                self.data_gen = False
                
        else:
            self.global_mapper_arr = global_guidance(self.agents_paths, self.grid.grid)
        
        self.agent_prev_coord = tuple(self.agents_paths[0][1:3])  # Take the first position of the path
        # The agent is modified to red
        self.init_arr[self.agent_prev_coord[0], self.agent_prev_coord[1]] = [255, 0, 0]  # Mark the agent's initial position in red
        self.init_arr[self.agents_paths[-1][1], self.agents_paths[-1][2]] = [0, 0, 255]  # Mark goal as blue

        # TODO: Implement a blue agent, which follows A* path, choosing idle action for every fifth time_idx. 
        # Additional reward for agent if stays close or surpasses blue agent. + reward dict element
        # Collision sholdn't be allowed with blue agent, because it will stay on A* path, for which agent recives icreased reward. (Potentially high collision risk.)

        self.time_idx = -1
        self.scenes = []
        self.leave_idx = -1
        
        # initialization state
        self.reset_state = self.agents_paths

        # initial distance
        start = tuple(self.reset_state[0][1:3])
        end = tuple(self.reset_state[-1][1:3])
        self.expected_cell = tuple(self.reset_state[1][1:3])
        self.dist = manhattan_distance(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
        # print(start, end, self.dist)

        # calc maximum steps
        if self.horizon != 'long':
            self.max_step = len(self.agents_paths) * 2

        self.agent_goal = end # Get the goal from the agent's path

        self.observation_history.clear()
        self.initial_random_steps = False

        # Take initial step to get the first real observation
        graphical_state, _, _, _, _ = self.step(4)  # Assuming 4 is a valid initial action

        self.first_reset = False
        return graphical_state, self.info

    def step(self, action):
        if len(self.init_arr) == 0:
            print("Run env.reset() first")
            return None, None, None, False

        conv, x, y = self.action_dict[action]

        self.direction = np.array([x, y])
        
        target_array = (2*self.local_fov, 2*self.local_fov, 4)
        
        self.time_idx += 1

        # save_array_as_png(self.init_arr, "before_step.png") 

        self.grid.ClearGrid(gridtype="grid_dynamic")
        if self.time_idx != 0 and (self.arrived % 2 != 1):
            self.grid.StepDynamicObstacles()
        self.grid.RenderAllObstacles(typ="dynamic")
        self.grid.RenderGrid()
        self.init_arr = GridToRGB(self.grid.grid)
        
        self.init_arr[self.agent_prev_coord[0], self.agent_prev_coord[1]] = [255, 0, 0]  # Mark the agent's initial position in red
        # self.init_arr[self.agents_paths[-1][1], self.agents_paths[-1][2]] = [0, 0, 255]  # Mark goal as blue

        # Update coordinates 
        self.local_obs, self.local_map, self.global_mapper_arr, done, trunc, self.info, rewards, \
        self.leave_idx, self.init_arr, new_agent_coord, self.dist, reached_goal, self.terminations, self.new_cost, self.penalty, self.expected_cell, self.was_reached = \
        update_agent(
            self.agents_paths, self.init_arr, self.time_idx,
            self.local_fov, self.global_mapper_arr, self.direction, self.agent_prev_coord,
            self.leave_idx, self.dist, self.agent_goal, self.terminations, self.info, self.grid_seq, self.orig_cost, self.new_cost, self.expected_cell, self.was_reached
        )

        # Generate data for ViNT 
        if self.allow_data_gen:
            self.datagen_new()

        # print(rewards)
        # Assuming self.init_arr is generated and is in uint8 format
        # save_array_as_png(self.init_arr, "after_step.png")  
        # save_array_as_png(local_obs, "local_obs.png")   
        # save_array_as_png(local_map, "global_map_part.png")   
        # save_array_as_png(self.global_mapper_arr, "global_map.png")   
        # save_array_as_png(self.init_arr, "after_step_after_dynamic.png") 

        self.steps += 1
        self.agent_prev_coord = new_agent_coord

        # Numerical State estimation using cost to goal - penalty for distancement from optimal route 
        self.state_estimate = ((self.orig_cost - self.new_cost) / self.orig_cost) - self.penalty

        # Update info
        self.info['steps'] += 1
        # self.info['path'].append((self.agent_prev_coord[0], self.agent_prev_coord[1]))
        self.info['reward'] += rewards

        if reached_goal == True:
            self.arrived += 0.5
            self.info['total_arrived'] += 0.5

        if done: 
            self.iter += 1 # Todo - iteration fix here

        # Check if there's global guidance in the local FOV
        if not self.has_global_guidance() and done == False and self.was_reached != True:
            trunc = True
            self.info['no_global_guidance'] = True
            self.terminations[1] += 1

        if self.steps > self.max_step and done == False:
            # print(f"Max steps reached with steps: {self.steps} for path length: {len(self.agents_paths)}, decay: {self.decay}")
            trunc = True
            self.info['R_max_step'] = True
            self.terminations[2] += 1

        if done or trunc: self.agent_last_coord = new_agent_coord

        combined_arr = np.array([])
        if len(self.local_obs) > 0:
            self.scenes.append(Image.fromarray(self.local_obs, 'RGB'))
            local_map = self.local_map.reshape(self.local_map.shape[0],self.local_map.shape[1], 1)
            combined_arr = np.dstack((self.local_obs, local_map))
            combined_arr = symmetric_pad_array(combined_arr, target_array, 255)
            combined_arr = combined_arr.reshape(1,1,combined_arr.shape[0], combined_arr.shape[1], combined_arr.shape[2])
        
        if len(combined_arr) > 0:
            if len(self.observation_history) < self.Nt:
                for _ in range(self.Nt):
                    self.observation_history.append(combined_arr)
                self.initial_random_steps = True
            else:
                # Remove the oldest observation and add the new one
                self.observation_history.popleft()
                self.observation_history.append(combined_arr)
    
        
        if self.initial_random_steps == False:
            # Return the single observation during initial steps
            return_values = (combined_arr, rewards, done, trunc, self.info)
        else:
            # Return the stacked state after we have enough observations
            stacked_state = self.get_stacked_state()
            return_values = (stacked_state, rewards, done, trunc, self.info)

        return return_values
        
    
    def get_stacked_state(self):
        # Ensure we have exactly Nt observations
        assert len(self.observation_history) == self.Nt, f"Expected {self.Nt} observations, but got {len(self.observation_history)}"
        
        # Stack the observations along the second axis (axis=1)
        stacked_state = np.concatenate(list(self.observation_history), axis=1)

        return stacked_state
    
    def f_action_space(self):
        # action space
        self.action_dict = {
            0:['up',0,1],
            1:['down',0,-1],
            2:['left',-1,0],
            3:['right',1,0],
            4:['idle',0,0]
        }
        return list(self.action_dict.keys())
    
    def action_mask(self, device):
        return self.get_action_mask(device)
    
    def get_action_mask(self, device, mask_type='single'):
        """
        Return a mask of valid actions, where 1 indicates a valid action and 0 indicates an invalid action.
        Args:
            device: The device on which the mask tensor is allocated.
            mask_type: The type of mask to use ('single' for single_cell_mask or 'double' for double_cell_mask).
        """
        mask = torch.ones(len(self.action_dict), dtype=torch.float32, device=device)

        # Get the current position of the agent
        agent_position = self.agent_prev_coord
        h, w = agent_position

        # Choose the appropriate mask function
        mask_function = self.single_cell_mask if mask_type == 'single' else self.double_cell_mask

        # Check each possible action and set mask to 0 for invalid actions
        if w < 0 or not mask_function(h, w - 1):  # up
            mask[0] = 0
        if w >= self.width or not mask_function(h, w + 1):  # down
            mask[1] = 0
        if h < 0 or not mask_function(h - 1, w):  # left
            mask[2] = 0
        if h >= self.height or not mask_function(h + 1, w):  # right
            mask[3] = 0

        if self.last_action == 4:
            mask[4] = 0
        else:  # Idle action is only valid if last 3 wasn't idle
            mask[4] = 1

        return mask

    def single_cell_mask(self, h, w):
    
        if h < 0 or h >= self.height or w < 0 or w >= self.width:
            # print(f"Position ({h}, {w}) is out of bounds")
            return False
        
        if (self.init_arr[h, w] == [0, 255, 101]).all():
            # print(f"Position ({h}, {w}) contains a dynamic obstacle")
            return False
        
        if (self.init_arr[h, w] == [203, 0, 255]).all():
            # print(f"Position ({h}, {w}) contains multiple dynamic obstacle")
            return False    
         
        if (self.init_arr[h, w] == [0, 0, 0]).all():
            # print(f"Position ({h}, {w}) contains a static obstacle")
            return False
        
        # If the goal was reached, check the Euclidean distance
        if self.was_reached:
            goal_h, goal_w = self.agent_goal  # Assuming self.agent_goal is a tuple (goal_h, goal_w)
            euclidean_distance = ((h - goal_h) ** 2 + (w - goal_w) ** 2) ** 0.5
            if euclidean_distance > 2:
                # print(f"Position ({h}, {w}) is farther than 2 units from the goal ({goal_h}, {goal_w})")
                return False
        
        # print(f"Position ({h}, {w}) is valid")
        return True
    
    def double_cell_mask(self, h, w):
        # Ensure the position is within bounds
        if h < 0 or h >= self.height or w < 0 or w >= self.width:
            return False

        # Define the range to check for neighboring cells (1 cell in each direction)
        neighboring_range = [-1, 0, 1]

        # Check for dynamic obstacles in the neighboring cells
        for dh in neighboring_range:
            for dw in neighboring_range:
                nh, nw = h + dh, w + dw  # Neighboring cell coordinates
                if 0 <= nh < self.height and 0 <= nw < self.width:  # Check bounds
                    if (self.init_arr[nh, nw] == [0, 255, 101]).all():  # Green dynamic obstacle
                        return False
                    if (self.init_arr[nh, nw] == [203, 0, 255]).all():  # Purple dynamic obstacle
                        return False

        # Check if the cell is a static obstacle
        if (self.init_arr[h, w] == [0, 0, 0]).all():
            return False

        # Check Euclidean distance from the goal if the goal was reached
        if self.was_reached:
            goal_h, goal_w = self.agent_goal  # Assuming self.agent_goal is a tuple (goal_h, goal_w)
            euclidean_distance = ((h - goal_h) ** 2 + (w - goal_w) ** 2) ** 0.5
            if euclidean_distance > 2:
                return False

        return True

    def generate_end_points_and_paths(self):
        """
        Generate destinations and routes for agents on the grid.
        """
        agents_coord = tuple([self.grid.GetRandomFreeCell()])
        start_end_coords = start_end_points(agents_coord, self.grid, self.width, self.np_random)

        self.agents_paths = []
        start = start_end_coords[0]
        end = start_end_coords[1]
        assert start != end, "Start and end coordinates cannot be indenticial"

        # Validate that the start and end coordinates are tuples of length 2
        if isinstance(start, (list, tuple)) and len(start) == 2 and isinstance(end, (list, tuple)) and len(end) == 2:
            self.grid.ResetDynamicObstacles()
            self.grid_seq = self.grid.GenerateGridSequence(self.frames, reset_after=True)
            # save_grid_with_name(self.grid_seq, "GridBasedPathPlanning/RL_Environment/ViNT_data", f"grid_seq_{self.episode_count}.npy")
            planner = TJPS(self.grid_seq, start, end, max_wait=4, log=0)
            self.plan = planner.plan()

            # If no valid plan is found, restart the process
            if self.plan is None or self.plan['path_extracted'] is None:
                print("No valid path found. Generating new coordinates.")
                self.generate_end_points_and_paths()  # Restart if path not found
                return

            # Path found: Process the plan
            self.orig_cost = self.plan['cost']
            self.new_cost = self.orig_cost
            path = self.plan['path_extracted'].astype(np.int64)
            self.agents_paths = path
            self.global_mapper_arr = global_guidance(self.agents_paths, self.grid.grid)

        else:
            raise ValueError("Start and end must be lists or tuples of length 2")

    def has_global_guidance(self):
        local_guidance = self.global_mapper_arr[
            max(0, self.agent_prev_coord[0] - self.local_fov):min(self.width, self.agent_prev_coord[0] + self.local_fov),
            max(0, self.agent_prev_coord[1] - self.local_fov):min(self.width, self.agent_prev_coord[1] + self.local_fov)
        ]
        
        # Check if there's any global guidance information (value less than 255) in the local observation
        has_guidance = np.any(local_guidance < 255)
        
        return has_guidance
    
    def render(self):
        if self.pygame_render:  # Check if rendering is enabled
            if self.screen is None:  # Initialize only if not already initialized
                pygame.init()
                print("Pygame screen constructed")
                self.screen = pygame.display.set_mode((800, 800))
                self.clock = pygame.time.Clock()
                pygame.display.set_caption(f"Warehouse Environment")

        if self.pygame_render == False:
            pygame.quit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Create a surface from the numpy array
        surf = pygame.surfarray.make_surface(self.init_arr)

        # Define the window size
        new_width, new_height = 800, 800  # Changable

        # Scale the surface to the size
        surf = pygame.transform.scale(surf, (new_width, new_height))

        # If the screen size doesn't match the new size, recreate it
        if self.screen.get_size() != (new_width, new_height):
            self.screen = pygame.display.set_mode((new_width, new_height))

        # Draw the path under the agent
        for i, (t, x, y) in enumerate(self.agents_paths):
            center_x = (x + 0.5) * new_width // self.init_arr.shape[1]
            center_y = (y + 0.5) * new_height // self.init_arr.shape[0]
            pygame.draw.circle(surf, (255, 0, 0), (center_x, center_y), 5)

        # Draw a purple square representing the local field of view around the agent
        if hasattr(self, 'agent_prev_coord'):
            # Calculate the top-left corner of the FOV square
            top_left_x = max(0, self.agent_prev_coord[0] - self.local_fov) * new_width // self.init_arr.shape[1]
            top_left_y = max(0, self.agent_prev_coord[1] - self.local_fov) * new_height // self.init_arr.shape[0]

            # Calculate the bottom-right corner of the FOV square
            bottom_right_x = min(self.init_arr.shape[1], self.agent_prev_coord[0] + self.local_fov) * new_width // self.init_arr.shape[1]
            bottom_right_y = min(self.init_arr.shape[0], self.agent_prev_coord[1] + self.local_fov) * new_height // self.init_arr.shape[0]

            # Draw the purple square
            pygame.draw.rect(surf, (128, 0, 128), pygame.Rect(top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y), 2)  # The last parameter is the thickness

        # Blit the scaled surface to the screen
        self.screen.blit(surf, (0, 0))

        # Display the global guidance map as a semi-transparent red overlay
        if hasattr(self, 'global_mapper_arr'):
            # Convert the global map to a surface
            guidance_surf = pygame.surfarray.make_surface(self.global_mapper_arr)
            guidance_surf = pygame.transform.scale(guidance_surf, (new_width, new_height))
            guidance_surf.set_alpha(128)  # Set transparency level
            self.screen.blit(guidance_surf, (0, 0))

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        self.clock.tick(20)  # 30 FPS

    def close(self):
        pygame.quit()
    
    def render_video(self, train_name, image_index):
        assert len(self.init_arr) != 0, "Run env.reset() before proceeding"
        # Get the most recent observation (last channel of the stacked state)
        img = Image.fromarray(self.init_arr, 'RGB')

        # Ensure the base directory 'training_images' exists
        base_dir = 'eval/training_images'
        os.makedirs(base_dir, exist_ok=True)

        # Create train_{train_index}_images directory if it does not exist
        train_dir = os.path.join(base_dir, f"train_name")
        os.makedirs(train_dir, exist_ok=True)

        # Save the image with a unique filename
        img_path = os.path.join(train_dir, f"{train_name}_{int(image_index)}.png")
        img.save(img_path)

    def render_gif(self):
        """
        Renders the current state of the environment in gif format for real-time visualization. 
        This method should be called after each step.
        """
        assert len(self.init_arr) != 0, "Run env.reset() before proceeding"
        
        # Convert the environment state to an image
        img = Image.fromarray(self.init_arr.astype('uint8'), 'RGB')
        
        # Resize the image if needed (optional, for better visualization)
        img = img.resize((200, 200), Image.NEAREST)
        
        # Convert PIL Image to numpy array
        frame = np.array(img)
        
        # Append the frame to our list of frames
        self.frames.append(frame)
        
        # Update the GIF file
        self._update_gif()

    def _update_gif(self):
        """
        Updates the GIF file with all frames collected so far.
        """
        # Save the frames as a GIF
        imageio.mimsave("data/g2rl.gif", self.frames, duration=0.5, loop=0)

    def create_scenes(self, path = "data/agent_locals.gif", length_s = 100):
        if len(self.scenes) > 0:
            self.scenes[0].save(path,
                 save_all=True, append_images=self.scenes[1:], optimize=False, duration=length_s*4, loop=0)
        else:
            pass

    def datagen_old(self, path = "GridBasedPathPlanning/RL_Environment/ViNT_data"):
        if self.data_gen == True: 
            # save the self.agents_paths as a numpy array
            np.save(f"{path}/tjps_paths/agent_path_map{self.map_idx}_env{self.idx}_iter{self.iter}.npy", self.agents_paths)   
            # save_numpy_to_txt('GridBasedPathPlanning/RL_Environment/ViNT_data/tjps_paths/agent_path_0.npy', 'GridBasedPathPlanning/RL_Environment/ViNT_data/tjps_paths/agent_path_0.txt')

            # Save the global guidance map as a PNG image
            global_map = binary_img_gray_white(self.global_mapper_arr, threshold=128, gray_value=128)
            global_map.save(f"{path}/global_guidances/global_map{self.map_idx}env{self.idx}_iter{self.iter}.png")
        else:
            # Save the whole coloured map (optional)
            #'''
            img = Image.fromarray(self.init_arr.astype('uint8'), 'RGB') 
            img.save(f"{path}/whole_maps_rgb/rgb_map{self.map_idx}_env_{self.idx}_iter{self.iter}_step{self.steps}.png")

            target_array = (2*self.local_fov, 2*self.local_fov, 3)
            local_grid_rgb = symmetric_pad_array(self.local_obs, target_array, 255)
            img = Image.fromarray(local_grid_rgb.astype('uint8'), 'RGB') 
            img.save(f"{path}/local_maps_rgb/rgb_local_map{self.map_idx}_env_{self.idx}_iter{self.iter}_step{self.steps}.png")
            #'''

            # Save the global guidance map as a PNG image
            local_map = binary_img_gray_white(self.local_map, threshold=128, gray_value=128)
            local_map.save(f"{path}/local_guidances/local_map{self.map_idx}_env{self.idx}_iter{self.iter}_step{self.steps}.png")
            
            # Save the static obstacles of the current local obs as a binary PNG image
            static_obs = binary_image_match_black(local_grid_rgb)
            static_obs.save(f"{path}/static_obs/static_obstacles_map{self.map_idx}_env{self.idx}_iter{self.iter}_step{self.steps}.png")

            # Save the dynamic obstacles of the current local obs as a binary PNG image
            dynamic_obs = binary_image_match_green(local_grid_rgb)
            dynamic_obs.save(f"{path}/dynamic_obs/dynamic_obstacles_map{self.map_idx}_env{self.idx}_iter{self.iter}_step{self.steps}.png")

    def datagen_new(self, path="GridBasedPathPlanning/RL_Environment/ViNT_data"):
        if self.first_reset != True and self.info['total_arrived'] < self.map_cap:
            if self.data_gen:
                # Create base folder (e.g. "GridBasedPathPlanning/RL_Environment/ViNT_data")
                base_dir = path
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)

                # Create subfolder with today's date (e.g., "2025-02-04")
                today = datetime.date.today().strftime("%Y-%m-%d")
                date_folder = os.path.join(base_dir, f"{today}_{self.name}")

                # Create the directory if it doesn't exist
                if not os.path.exists(date_folder):
                    os.makedirs(date_folder)

                # Create subfolder for the current iteration and environment (e.g., "iter3_env1")
                iter_env_folder = os.path.join(date_folder, f"map{self.map_idx}_iter{self.iter}_env{self.idx}")
                if not os.path.exists(iter_env_folder):
                    os.makedirs(iter_env_folder)

                # Assume self.agents_paths is an array-like object where each row is [time, x, y].
                # Convert it to a NumPy array and extract the x,y coordinates.
                positions_full = np.array(self.agents_paths)  # shape: (T, 3)
                pos_xy = positions_full[:, 1:3]  # shape: (T, 2)
                T = pos_xy.shape[0]

                # Compute yaw for each timestep from the differences between consecutive positions.
                # For each timestep t >= 1, yaw[t] = arctan2(delta_y, delta_x).
                # For the first timestep, we simply use the yaw computed for the second timestep.
                if T < 2:
                    yaw_array = np.zeros((T,))
                else:
                    deltas = np.diff(pos_xy, axis=0)  # differences between consecutive positions
                    computed_yaw = np.arctan2(deltas[:, 1], deltas[:, 0])
                    computed_yaw = (computed_yaw + 2 * np.pi) % (2 * np.pi)
                    for i in range(1, len(pos_xy)):
                        delta = pos_xy[i] - pos_xy[i-1]
                        # print(f"Delta: {delta}, Computed Yaw: {computed_yaw[i-1]}")
                        # print(f"From: {pos_xy[i]}, To: {pos_xy[i-1]}")
                    # Prepend the first computed yaw so that yaw_array has length T.
                    yaw_array = np.insert(computed_yaw, 0, computed_yaw[0])

                # Build the trajectory dictionary in the format expected by VisualNav-Transformer.
                traj_data = {"position": pos_xy, "yaw": yaw_array}

                # Save the trajectory dictionary as a pickle file using the highest protocol.
                traj_pickle_file = os.path.join(iter_env_folder, "traj_data.pkl")
                with open(traj_pickle_file, "wb") as f:
                    pickle.dump(traj_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                today = datetime.date.today().strftime("%Y-%m-%d")
                date_folder = os.path.join(path, f"{today}_{self.name}")

                # Create subfolder for the current iteration and environment (e.g., "iter3_env1")
                iter_env_folder = os.path.join(date_folder, f"map{self.map_idx}_iter{self.iter}_env{self.idx}")
                if not os.path.exists(iter_env_folder):
                    os.makedirs(iter_env_folder) 

                # Save other images if data_gen is False.
                target_array = (2 * self.local_fov, 2 * self.local_fov, 3)
                local_grid_rgb = symmetric_pad_array(self.local_obs, target_array, 255)
                img = Image.fromarray(local_grid_rgb.astype('uint8'), 'RGB')
                # Save the image with error handling
                img_path = f"{date_folder}/map{self.map_idx}_iter{self.iter}_env{self.idx}/rgb_local_map{self.map_idx}_step{self.steps}.png"
                try:
                    img.save(img_path)
                    time.sleep(0.1)
                    # print(f"Saved image: {img_path}")
                except Exception as e:
                    print(f"Error saving {img_path}: {e}")


'''
env = WarehouseEnvironment(pygame_render=True)
state, info = env.reset() # image of first reset

rgb_array = env.init_arr

# Reshape the array to be a list of RGB values (each pixel is a 3-tuple)
reshaped_rgb = rgb_array.reshape(-1, 3)

# Find the unique RGB values
unique_colors = np.unique(reshaped_rgb, axis=0)

# Print each unique color
print("Unique RGB colors found in the array:")
for color in unique_colors:
    print(color)

# (batch_size, time_dim, width, height, chanels) - (1,1,30,30,4)
print(state.shape)
print(env.init_arr.shape)

while(env.pygame_render==True):
    env.render()
    env.step(1)
#'''

