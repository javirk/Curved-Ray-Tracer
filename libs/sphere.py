import torch
import libs.utils as u
from libs.utils import dev


class Sphere:
    def __init__(self, center, radius, material):
        super().__init__()
        self.center = center.to(dev)
        self.radius = radius
        self.material = material

    def hit(self, r, distance, color):
        intersections = distance < self.radius
        if intersections.any():
            color = color.detach().clone()
            outward_normal = (r.pos - self.center) / self.radius
            scattered_pos, scattered_vel = self.material.scatter(r, outward_normal)
            r.update(intersections, scattered_pos, scattered_vel)
            color += torch.pow(self.material.albedo, r.depth)

        return r, color, intersections

    def intersect(self, r):
        oc = r.pos - self.center
        distances = torch.sqrt(u.dot(oc, oc))
        return distances
