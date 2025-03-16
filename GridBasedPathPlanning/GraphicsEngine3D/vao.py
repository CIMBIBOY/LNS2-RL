from vbo import VBO
from ibo import IBO
from shader_program import ShaderProgram


class VAO:
    def __init__(self, ctx, data):
        self.ctx = ctx
        self.vbo = VBO(ctx, data) # data added
        self.ibo = IBO(ctx)
        self.program = ShaderProgram(ctx)
        self.vaos = {}

        # cube static vao
        self.vaos['cubeStatic'] = self.get_vao(
            program = self.program.programs['cubeStatic'],
            vbo = [self.vbo.vbos['cube'],
                   self.vbo.vbos['cubeStaticInstance']],
            ibo = self.ibo.ibos['cube'])

        # cube dynamic vao
        self.vaos['cubeDynamic'] = self.get_vao(
            program = self.program.programs['cubeDynamic'],
            vbo = [self.vbo.vbos['cube'],
                   self.vbo.vbos['cubeDynamicInstance']],
            ibo = self.ibo.ibos['cube'])

    def get_vao(self, program, vbo, ibo=None):
        
        vao = self.ctx.vertex_array(program = program, 
                                    content = [(vbo[i].vbo, vbo[i].format, *vbo[i].attribs) for i in range(len(vbo))],
                                    index_buffer = ibo.ibo if ibo is not None else None)
        return vao

    def destroy(self):
        self.vbo.destroy()
        self.program.destroy()