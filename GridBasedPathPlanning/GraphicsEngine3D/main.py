import pygame as pg
import moderngl as mgl
import os, sys
from datetime import datetime

from clock import Clock
from model import *
from camera import Camera
from light import Light
from mesh import Mesh
from scene import Scene
from data import Data

class GraphicsEngine:
    def __init__(self, win_size=(1920, 1080)):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        # detect and use existing opengl context
        self.ctx = mgl.create_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
        
        self.clock  = Clock()       # clock / timing
        self.data   = Data(self, subfolder='demo_5drones') # 3D_multistep
        self.light  = Light()
        self.camera = Camera(self, position=(2,0.5,0), yaw=180,pitch=-20)
        self.mesh   = Mesh(self)
        self.scene  = Scene(self)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.mesh.destroy()
                pg.quit()
                sys.exit()
            elif (event.type == pg.ACTIVEEVENT and event.gain == 1 and event.state == 2): # Windows regains focus
                pg.event.set_grab(True)
                pg.mouse.set_visible(False)
            elif (event.type == pg.KEYDOWN and event.key == pg.K_r): # R: Reset the animation
                self.clock.time_reset = self.clock.time
            elif (event.type == pg.KEYDOWN and event.key == pg.K_p): # P: Screenshot
                self.take_screenshot()
            elif (event.type == pg.KEYDOWN and event.key == pg.K_i): # I: Interpolate camera movement
                self.camera.camera_interp = not self.camera.camera_interp

    def render(self):
        self.ctx.clear(color=(0.08, 0.16, 0.18))    # clear framebuffer
        self.scene.render()                         # render scene
        pg.display.flip()                           # swap buffers

    def run(self):
        while True:
            self.clock.update_time()
            self.check_events()
            self.camera.update()
            self.render()
            self.clock.update_delta_time()
            pg.display.set_caption(f'GraphicsEngine FPS: {self.clock.get_FPS():.0f}, t: {self.clock.time_animation:.1f}, {self.camera.get_camera_str()}')

    def take_screenshot(self):
        data = self.ctx.screen.read()   # Read the pixel data from the framebuffer
        image = pg.image.frombytes(data, self.WIN_SIZE, 'RGB',True)
        pg.image.save(image,f'GridBasedPathPlanning/Data/Screenshots/GraphicsEngine3D_{datetime.today().strftime('%Y%m%d_%H%M%S')}.png')

if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()
