import torch
import libs.utils as u


class World:
    def __init__(self, device='cpu'):
        self.objects = []
        self.lights = []
        self.t_max = torch.tensor(1.e5, device=device)

    def add(self, o):
        self.objects.append(o)

    def clear(self):
        self.objects = []

    def add_light(self, l):
        self.lights.append(l)

    def hit(self, r_in):
        distances = [obj.intersect(r_in) for obj in self.objects]  # hit should return the distance to the object
        nearest = torch.min(torch.stack(distances, dim=0), dim=0).values

        return distances, nearest

    def hit_shadows(self, r_in):
        intersections = [obj.intersect_straight(r_in, self.t_max) for obj in self.objects] # hit should return t
        nearest = torch.min(torch.stack(intersections, dim=0), dim=0).values

        return intersections, nearest
