import torch
from libs.utils import dev, FARAWAY
from einops import repeat
import libs.utils as u


class Rays:
    def __init__(self, origin, directions):
        if len(origin) == 1:
            self.origin = repeat(origin, 'c -> copy c', copy=directions.shape[0])
        else:
            self.origin = origin

        self.pos = (directions + origin).to(dev)
        self.vel = u.unit_vector(directions.float()).to(dev)
        self.depth = torch.zeros((self.n_rays(), 1)).to(dev)

    def n_rays(self):
        return self.pos.shape[0]

    def x(self):
        return self.pos[:, 0]

    def y(self):
        return self.pos[:, 1]

    def z(self):
        return self.pos[:, 2]

    def point_at(self, t):
        return self.origin + self.vel * t

    def update(self, cond, pos, vel):
        self.pos = torch.where(cond, pos, self.pos)
        self.vel = torch.where(cond, vel, self.vel)
        depth_1 = self.depth + 1
        self.depth = torch.where(cond, depth_1, self.depth)
