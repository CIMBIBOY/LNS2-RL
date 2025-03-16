import pygame as pg
import moderngl as mgl
import os

class Texture:
    def __init__(self, ctx):
        self.ctx = ctx
        self.textures = {}

        parent_dir_path = os.path.dirname(__file__)

        #self.textures['cube'] = self.get_texture(path='GraphicsEngine3D/textures/img.png')
        #self.textures['cat'] = self.get_texture(path=os.path.join(parent_dir_path, 'objects/cat/20430_cat_diff_v1.jpg'))
        self.textures['drone'] = self.get_texture(path=os.path.join(parent_dir_path, 'objects/drone/MQ-9_Black.png'))

    def get_texture(self, path):
        texture = pg.image.load(path).convert()
        texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
        texture = self.ctx.texture(size=texture.get_size(),
                                   components=3,
                                   data=pg.image.tostring(texture, 'RGB'))
        # mipmaps
        texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        texture.build_mipmaps()
        # AF
        texture.anisotropy = 32.0
        return texture

    def destroy(self):
        [tex.release() for tex in self.textures.values()]