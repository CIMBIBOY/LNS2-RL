import os, sys
import pygame
import numpy as np
import threading
from datetime import datetime

from GridBasedPathPlanning.Environment.GridMapEnv_Utils import getColorMap

class GraphicsEngine2D():
    def __init__(self, grid_seq:np.ndarray=None, extracted_path:np.ndarray=None, FPS = 30, FPS_animation:int = 30, win_size:tuple=(1000,1000)):
        pygame.init()
        
        # SCREEN
        self.WIN_SIZE = win_size
        self.cmap = getColorMap()
        self.surf = None

        self.render_event = threading.Event()

        if grid_seq is not None:
            self.grid_seq = grid_seq
            self.grid_shape = grid_seq[0].shape
            self.pix = (self.WIN_SIZE[0] / self.grid_shape[0] )  # The size of a single grid square in pixels
            self.render_event.set()
        else:
            self.render_event.clear()

        if extracted_path is not None:
            self.path = extracted_path[:,1:]
        
        

        pygame.init()
        self.display = pygame.display.set_mode(win_size)
        pygame.display.set_caption('GraphicsEngine2D')

        # TIME
        self.clock = pygame.time.Clock()
        self.FPS = FPS
        self.FPS_animation = FPS_animation
        self.time = 0
        self.delta_time = 0
        self.time_reset = 0
        self.frame_index_prev = -1

    def update_data(self, grid_seq:np.ndarray=None, extracted_path:np.ndarray=None):
        self.grid_seq = grid_seq
        self.grid_shape = grid_seq[0].shape
        self.pix = (self.WIN_SIZE[0] / self.grid_shape[0] )  # The size of a single grid square in pixels
        self.path = extracted_path[:,1:]

        self.time_reset = self.time


    def render(self):
        frame_index = min(int(self.time_animation * self.FPS_animation), self.grid_seq.shape[0]-1)
        frame_index_path = min(int(self.time_animation * self.FPS_animation), self.path.shape[0]-1)

        surf_array = self.grid_seq[frame_index]
        surf_array = np.where(surf_array == np.inf, 1, surf_array)

        self.surf = pygame.surfarray.make_surface((np.fliplr(surf_array*255).astype(int)))
        self.surf = pygame.transform.scale(self.surf,self.WIN_SIZE)
        self.surf.set_palette(self.cmap)

        self.surf = self.surf.convert_alpha()
        
        if self.path is not None:
            # PATH
            for i in range(self.path.shape[0]-1):
                pygame.draw.line(self.surf, (240,120,140),
                    ((self.path[i][0] + 0.5)*self.pix, (self.grid_shape[1]-self.path[i][1]-  0.5)*self.pix),
                    ((self.path[i+1][0]+0.5)*self.pix, (self.grid_shape[1]-self.path[i+1][1]-0.5)*self.pix), width=4)
                
                # Draw a small dot on the path if the agent has waited
                if np.all(self.path[i] == self.path[i+1]):
                    pygame.draw.circle(self.surf, (240,120,140),
                    ((self.path[i][0]+0.5)*self.pix, (self.grid_shape[1]-(self.path[i][1]+0.5))*self.pix), self.pix)
                    
            # AGENT
            pygame.draw.circle(self.surf, (255, 0, 0), 
                ((self.path[frame_index_path][0]+0.5)*self.pix, (self.grid_shape[1]-(self.path[frame_index_path][1]+0.5))*self.pix), self.pix)
                

        self.display.blit(self.surf,(0,0))   # (0,0) -> center in window
        pygame.display.update()
        self.clock.tick(self.FPS)

        
        if frame_index > self.frame_index_prev:
            #pygame.image.save(self.surf,f'Screenshots/GraphicsEngine2D_{(self.time*1000):.0f}.png')
            self.frame_index_prev = frame_index
        

    def check_events(self):
        for event in pygame.event.get():
            if  event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()
                    return
            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_r): # R: Reset the animation
                self.time_reset = self.time
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                os.makedirs('GridBasedPathPlanning/Data/Screenshots/', exist_ok=True)
                pygame.image.save(self.surf,f'GridBasedPathPlanning/Data/Screenshots/GraphicsEngine2D_{datetime.today().strftime('%Y%m%d_%H%M%S')}.png') # P: Screenshot

    def update_time(self):
        self.time = pygame.time.get_ticks() * 0.001 # sec
        self.time_animation = self.time - self.time_reset

    def update_delta_time(self):
        self.delta_time = self.clock.tick(self.FPS)

    def run(self):
        while True:
            self.render_event.wait()
            self.update_time()
            self.check_events()
            if self.render_event.is_set():
                self.render()
            self.update_delta_time()
