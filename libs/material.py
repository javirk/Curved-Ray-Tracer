from libs.utils import random_on_unit_sphere_like, dot, unit_vector
from libs.rays import Rays
import torch

class Material:
    def __init__(self, mat, attenuation):
        self.material = mat
        if attenuation.dtype == torch.int64:
            self.albedo = attenuation / 256.
        else:
            self.albedo = attenuation

        if self.material == 'lambertian':
            self.scatter = self.scatter_lambertian
        elif self.material == 'metal':
            self.scatter = self.scatter_metal
        elif self.material == 'light':
            self.scatter = self.scatter_light
        else:
            raise ValueError(self.material + ' unknown material')


    def scatter_lambertian(self, r_in, normal):
        scatter_direction = normal + random_on_unit_sphere_like(r_in.vel, device=r_in.dev) #+ r_in.vel # This is actually a velocity
        scatter_pos = r_in.pos + normal * 1e-4 # Normal goes outward, this is to prevent the next step to hit the sphere again
        return scatter_pos, scatter_direction

    def scatter_metal(self, r_in, normal):
        reflected = self._reflect(unit_vector(r_in.vel, dim=1), normal)
        scatter_pos = r_in.pos + normal * 0.1
        return scatter_pos, reflected

    def scatter_light(self, r_in, normal):
        return r_in.pos, r_in.vel

    @staticmethod
    def _reflect(v, n):
        return v - 2 * dot(v, n) * n