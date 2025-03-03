import pygame as pg

class Clock:
    def __init__(self):
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0
        self.time_animation = 0
        self.time_reset = 0

        self.FPS = 30
        self.FPS_animation = 15
        self.delay = 0.3

    def update_time(self):
        self.time = pg.time.get_ticks() * 0.001 # sec

        self.time_animation = max(0, self.time - self.time_reset - self.delay)

    def update_delta_time(self):
        self.delta_time = self.clock.tick(self.FPS)

    def get_FPS(self):
        return self.clock.get_fps()