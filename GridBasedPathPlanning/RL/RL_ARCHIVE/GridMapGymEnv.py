import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import pygame

from GridBasedPathPlanning.Environment.GridMapEnv import GridMapEnv


class GridMapGymEnv(gym.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

	def __init__(self, params, render_mode=None):
		self.size = params['GridSize']      		# The size of the square grid
		self.window_size = 1000  					# The size of the PyGame window
		
		self.GridMapEnv = GridMapEnv(self.size,self.size)	# NEEDS TO BE INITIALIZED TO USE IN THE RESET METHOD
		#self.obs_window_size = 9					# Uneven number !!!
		#self.padded_grid = np.ones((self.size + self.obs_window_size - 1,self.size + self.obs_window_size - 1),dtype=int)

		self.episode_num = 0
		#self.collision_num = 0
		self.path_length = 0
		#self.optimal_path_from_start_position = []
		#self.optimal_path_from_current_position = []
		#self.optimal_path_from_prev_position = []

		self._agent_target_dist_init = 0
		self._agent_target_dist_min = 0

		# Observations are dictionaries with the agent's and the target's location.
		# Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
		self.observation_space = spaces.Dict(
			{
				"agent":   spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
				"target":  spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
				"gridmap": spaces.Box(0, 1, shape=(self.size,self.size), dtype=float)
			}
		)

		self.action_space = spaces.Discrete(9)

		# The following dictionary maps abstract actions from `self.action_space` to
		# the direction we will walk in if that action is taken.
		self._action_to_direction = {
			0: np.array([1, 0]),	# X: +1 Y: 0
			1: np.array([1, 1]),	
			2: np.array([0, 1]),
			3: np.array([-1, 1]),
			4: np.array([-1, 0]),
			5: np.array([-1, -1]),
			6: np.array([0, -1]),
			7: np.array([1, -1]),
			8: np.array([0, 0])		# WAIT action
		}

		self.rewards = {
			"reaching_goal": 1000,
			"step_in_optimal_direction": 0,	# NOT IMPLEMENTED
			"time_passing": -0.2,
			"timeout": -50,
		}

		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode

		# If human-rendering is used, `self.window` will be a reference
		# to the window that we draw to. `self.clock` will be a clock that is used
		# to ensure that the environment is rendered at the correct framerate in human-mode.
		self.window = None
		self.clock = None

		self.cmap = None

	def _get_obs(self):
		"""
		near_grid = np.zeros((3,3), dtype=int)
		x = self._agent_location[0]
		y = self._agent_location[1]
		obs_window_size_side = int(((self.obs_window_size-1)/2))

		self.padded_grid[obs_window_size_side:self.size + obs_window_size_side ,
			  	    obs_window_size_side:self.size + obs_window_size_side] = self.GridMapEnv.grid
		near_grid = self.padded_grid[x:x + 1 + obs_window_size_side*2,y:y + 1 + obs_window_size_side*2]
		 "gridmap": near_grid				# np array (obs_window_size,obs_window_size)
		"""

		return {
			"agent":   self._agent_location,	# np array (2,)
		  	"target":  self._target_location,	# np array (2,)
			"gridmap": self.GridMapEnv.grid			# np array (size,size)
		}

	def _get_info(self):
		return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
				"path_length":   self.path_length}

	def reset(self, seed=None, options=None):
	
		super().reset(seed=seed)	# We need the following line to seed self.np_random
		##########	RESET VARIABLES	##########	
		
		self.collision_num = 0
		self.path_length = 0
		"""
		self.optimal_path_from_start_position = []
		self.optimal_path_from_current_position = []
		self.optimal_path_from_prev_position = []
		"""

		##########	CREATE THE NEW MAP	##########

		# TOTALLY RESET THE ENVIRONMENT AFTER xxx EPISODES
		if self.episode_num % 1000 == 0:
			# Create STATIC obstacles
			self.GridMapEnv.CreateObstacle(typ='static',shape = 'circle',data = {'p':np.array([5,5]),'d':20,'alpha':0.95,'gradient':0.7})
			# Create DYNAMIC obstacles
			self.GridMapEnv.CreateObstacle(typ='dynamic',shape='circle',data={'p': np.array([38,24]),'d':40,'vel':np.array([0.4,0.8]),'alpha': 0.8,'gradient':0.7})

			self.GridMapEnv.RenderAllObstacles(typ='static')
			self.GridMapEnv.RenderAllObstacles(typ='dynamic')
			self.GridMapEnv.CreateBorders()
			self.GridMapEnv.RenderGrid()
		else:
			# STEP WITH THE OBSTACLES AFTER xxx EPISODES
			if self.episode_num % 5 == 0:
				self.GridMapEnv.ClearGrid(gridtype='grid_dynamic')
				self.GridMapEnv.StepDynamicObstacles()
				self.GridMapEnv.RenderAllObstacles(typ='dynamic')
				self.GridMapEnv.RenderGrid()


		"""
		start = self.GridMapEnv.GetRandomFreeCell()
		while np.array_equal(start, (goal :=  self.GridMapEnv.GetRandomFreeCell())):
			pass
		#self.optimal_path_from_start_position = self.GridMapEnv.FindPathAStar(start=start, goal=goal)
		self.optimal_path_from_start_position = self.GridMapEnv.FindPathJPS(start=start, goal=goal)
		if (self.optimal_path_from_start_position != None):		# Exit condition
			break
		"""

		print("Start: ", start := np.array([5,5]))
		print("Goal: ",  goal := np.array([95,95]))

		self._agent_location = start
		self._target_location = goal
		self._agent_target_dist_init = np.linalg.norm(self._agent_location - self._target_location)
		self._agent_target_dist_min = self._agent_target_dist_init


		##########	GET OBS AND INFO	  ##########

		observation = self._get_obs()
		info = self._get_info()

		self.episode_num += 1

		if self.render_mode == "human":
			self._render_frame()
		elif self.render_mode == None:
			print(f"Epsiode number: {self.episode_num}")

		return observation, info

	def step(self, action):

		reward = 0
		done = False

		"""
		# Determine the optimal step for the previous agent location before updating the current location
		#prev_agent_location = self._agent_location			# Works in the first step
		if self.optimal_path_from_prev_position == []: 		# First step of the agent
			self.optimal_path_from_prev_position = self.optimal_path_from_start_position
		else:
			self.optimal_path_from_prev_position = self.optimal_path_from_current_position
		"""

		# Update the location of the agent to the current step
		move_direction = self._action_to_direction[int(action)]		# Map the action to the direction we walk in
		self._agent_location = np.clip(	self._agent_location + move_direction, 0, self.size - 1	) # Make sure we don't leave the grid

		# An episode is done if the agent has reached the target or commits suicide by getting stuck
		if(np.array_equal(self._agent_location, self._target_location)):
			reward += self.rewards["reaching_goal"]
			done = 	True

		"""
		# The new optimal path is only calculated if the agent hasnt reached its goal
		# If there is no path we consider the task as DONE and give a negative reward
		if not done:
			#self.optimal_path_from_current_position = self.GridMapEnv.FindPathAStar(self._agent_location,self._target_location)
			self.optimal_path_from_current_position = self.GridMapEnv.FindPathJPS(self._agent_location,self._target_location)
			if self.optimal_path_from_current_position == None:	# If there is no possible path from the current position to the target
				reward += self.rewards["suicide"]
		"""

		# Motion cost:
		if (action == 8): 		# WAIT
			move_cost = 1
		elif (action%2 == 1):	# Diagonal
			move_cost = 1.4142
		else: 
			move_cost = 1		# Horizontal or vertical

		# Cost of moving weighted by the grid cost
		# Cost of action = move_cost + move_cost*grid_weight
		reward +=  -move_cost * (1 + 10*self.GridMapEnv.grid[tuple(self._agent_location)])
		
		# Penalize the agent for every passing timestep
		reward += self.rewards["time_passing"]

		# Reward the agent for getting closer to the goal
		_agent_target_dist_curr = np.linalg.norm(self._agent_location - self._target_location)


		self._agent_target_dist_min
		if _agent_target_dist_curr < self._agent_target_dist_min:
			self._agent_target_dist_min = _agent_target_dist_curr
			reward += 5 + 5 * (self._agent_target_dist_init - _agent_target_dist_curr)/self._agent_target_dist_init


		# Penalize the agent for reaching the timeout
		self.path_length += 1		# Used for external benchmarking
		if self.path_length > (self.size**3):
			reward += self.rewards["timeout"]
			done = True

		if self.render_mode == "human":
			print(reward)


		"""
		# REWARD only if the agent is alive
		if not done:
			# Step in the optimal direction
			if self.optimal_path_from_prev_position is not None:
				optimal_step = np.array(self.optimal_path_from_prev_position[1])
				if np.array_equal(self._agent_location, optimal_step):
					reward += self.rewards["step_in_optimal_direction"]	# Reward the agent if it has stepped in the optimal direction
		"""

		###############################################################	

		observation = self._get_obs()
		info = self._get_info()

		if self.render_mode == "human":
			self._render_frame()

		return observation, reward, done, False, info
	
	def render(self):
		if self.render_mode == "rgb_array":
			return self._render_frame()

	def _render_frame(self):
		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode((self.window_size, self.window_size))
		if self.clock is None and self.render_mode == "human":
			self.clock = pygame.time.Clock()

		if self.cmap is None and self.render_mode == "human":
			cmap = mpl.colormaps['magma'].resampled(255)
			cmap = cmap(np.linspace(0, 1, 256))*255
			cmap = np.delete(cmap,-1,axis=1).tolist()

		surf = pygame.surfarray.make_surface((np.fliplr(self.GridMapEnv.grid*255).astype(int)))
		surf = pygame.transform.scale(surf,(1000,1000))
		surf.set_palette(cmap)

		# Draw the agent
		canvas = pygame.surface.Surface((self.window_size, self.window_size), pygame.SRCALPHA, 32)
		pix_square_size = (self.window_size / self.size	)  # The size of a single grid square in pixels
		canvas = canvas.convert_alpha()
		pygame.draw.circle(canvas, (0, 255, 0),
					      (np.array([self._agent_location[0]+0.5, self.size-self._agent_location[1]-0.5])) * pix_square_size,
						  pix_square_size/2)


		"""
		canvas = pygame.Surface((self.window_size, self.window_size))
		canvas.fill((255, 255, 255))
		pix_square_size = (self.window_size / self.size	)  # The size of a single grid square in pixels

		# Draw the optimal_path_from_start_position
		for i in range(len(self.optimal_path_from_start_position)-1):
			pygame.draw.line(
				canvas,
				(0,125,0),
				((self.optimal_path_from_start_position[i][0] +   0.5)*pix_square_size,
	 			 (self.optimal_path_from_start_position[i][1] +   0.5)*pix_square_size),
				((self.optimal_path_from_start_position[i+1][0] + 0.5)*pix_square_size,
	 			 (self.optimal_path_from_start_position[i+1][1] + 0.5)*pix_square_size),
				width=5,
			)
		"""



		if self.render_mode == "human":
			# The following line copies our drawings from `canvas` to the visible window
			#self.window.blit(canvas, canvas.get_rect())
			#pygame.event.pump()
			#pygame.display.update()

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.display.quit()
					pygame.quit()

			self.window.blit(surf, (0,0))   # (0,0) -> center in window
			self.window.blit(canvas, canvas.get_rect())

			pygame.display.update()     # Allows to change only part of the scrren

			# We need to ensure that human-rendering occurs at the predefined framerate.
			# The following line will automatically add a delay to keep the framerate stable.
			self.clock.tick(self.metadata["render_fps"])
		

		else:  # rgb_array
			#return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
			pass

		#im = plt.imshow(np.transpose(self.GridMapEnv.grid), cmap='magma', vmin=0, vmax=1, aspect='equal',origin="lower")
		#im = plt.plot([self._agent_location[0]], [self._agent_location[1]], 'go-')
		#plt.show()

	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()


