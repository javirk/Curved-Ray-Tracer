import torch
from libs.utils import FARAWAY
from einops import repeat, rearrange
import libs.utils as u


class Rays:
    def __init__(self, origin, directions, device):
        if len(origin) == 1:
            self.origin = repeat(origin, 'c -> copy c', copy=directions.shape[0])
        else:
            self.origin = origin

        self.pos = (directions + origin).to(device)
        self.vel = u.unit_vector(directions.float(), dim=1).to(device)
        self.dev = device

    @classmethod
    def from_shadow(cls, o, d, device='cpu'):
        r = cls(o - d, d, device)
        return r

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
