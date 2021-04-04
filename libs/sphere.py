import torch
import libs.utils as u
from libs.utils import dev


class Sphere:
    def __init__(self, center, radius, material):
        super().__init__()
        self.center = center.to(dev)
        self.radius = radius
        self.material = material

    def hit(self, r, color):
        color = color.detach().clone()
        oc = r.pos - self.center
        intersections = torch.sqrt(u.dot(oc, oc)) < self.radius
        if intersections.any():
            outward_normal = oc / self.radius
            scattered_pos, scattered_vel = self.material.scatter(r, outward_normal)
            r.update(intersections, scattered_pos, scattered_vel)
            color += torch.pow(self.material.albedo, r.depth)

        return r, color, intersections
