from model import *
import copy

class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        plans = self.app.data.plans

        self.add_object(CubeStatic(app, rot=(90,0,180)))    # total instanced cube size: 2x2x2
        
        if plans:
            #plans = [plans[0]]

            for plan_idx in range(len(plans)):

                #self.add_object(Spline(app, vao_name='spline1'+str(plan_idx), plan=plans[plan_idx], path_name='path_extracted', color = glm.vec4([31/255,119/255,180/255,1])))
                #self.add_object(Spline(app, vao_name='spline2'+str(plan_idx), plan=plans[plan_idx], path_name='path_corrected', color = glm.vec4([44/255,160/255,44/255,1])))
                if 'path_interp_BSpline' in plans[plan_idx] and 'path_interp_MinimumSnapTrajectory' not in plans[plan_idx]:
                    self.add_object(DroneOBJ(app, vao_name='drone'+str(plan_idx), plan = plans[plan_idx]))
                    self.add_object(Spline(app, vao_name='spline3'+str(plan_idx), plan=plans[plan_idx], path_name='path_interp_BSpline', color = glm.vec4([0.8,0.2,0.2,1])))
                if 'path_interp_MinimumSnapTrajectory' in plans[plan_idx]:
                    self.add_object(DroneSTL(app, vao_name='drone'+str(plan_idx), plan = plans[plan_idx]))
                    self.add_object(Spline(app, vao_name='spline4'+str(plan_idx), plan=plans[plan_idx], path_name='path_interp_MinimumSnapTrajectory', color = glm.vec4([0.8,0.2,0.2,1]))) # [1,0.9,0.2,1]

        self.add_object(CubeDynamic(app, rot=(90,0,180)))   # total instanced cube size: 2x2x2
        #self.add_object(CoordSys(app, vao_name='coordsys', pos=(0,0.4,0), rot=(90,0,180), scale=(0.1,0.1,0.1)))


    def render(self):
        for obj in self.objects:
            obj.render()