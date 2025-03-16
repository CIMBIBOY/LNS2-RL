import os

class ShaderProgram:
    def __init__(self, ctx):
        self.ctx = ctx
        self.programs = {}
        self.programs['default'] = self.get_program('default')
        self.programs['cube'] = self.get_program('cube')
        self.programs['cubeStatic'] = self.get_program('cubeStatic')
        self.programs['cubeDynamic'] = self.get_program('cubeDynamic')
        #self.programs['default_STL'] = self.get_program('default_STL')
        self.programs['spline'] = self.get_program('spline')
        self.programs['coordsys'] = self.get_program('coordsys')


    def get_program(self, shader_program_name):

        dir = os.path.dirname(os.path.abspath(__file__)) # GraphicsEngine3D folder
       

        with open(dir + f'/shaders/{shader_program_name}.vert') as file:
            vertex_shader = file.read()

        with open( dir + f'/shaders/{shader_program_name}.frag') as file:
            fragment_shader = file.read()

        program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return program

    def destroy(self):
        [program.release() for program in self.programs.values()]
