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
            outward_normal = (r.pos - self.center) / self.radius
            scattered_pos, scattered_vel = self.material.scatter(r, outward_normal)
            r.update(intersections, scattered_pos, scattered_vel)
            color += torch.pow(self.material.albedo, r.depth)
            # color += self.material.albedo * r.depth

        return r, color, intersections

    def get_color(self, r_in, t, world, depth=1):
        color = torch.zeros_like(r_in.directions, device=dev)
        p = r_in.point_at(t)
        outward_normal = (p - self.center) / self.radius
        r_scattered = self.material.scatter(outward_normal, p, r_in=r_in)
        if depth > 0:
            color += self.material.albedo * r_scattered.trace(world, depth - 1)

        return color

