import torch
import libs.utils as u
# from libs.utils import dev
from libs.rays import Rays


class Sphere:
    def __init__(self, center, radius, material, device):
        self.center = center.to(device)
        self.radius = radius
        self.material = material

    def hit(self, r, distance, color, world):
        intersections = distance < self.radius
        if intersections.any():
            color = color.detach().clone()
            outward_normal = (r.pos - self.center) / self.radius
            scattered_pos, scattered_vel = self.material.scatter(r, outward_normal)
            r.update(intersections, scattered_pos, scattered_vel)

            r_shadow = Rays.from_shadow(r.pos, world.light - r.pos, r.dev)
            distances, nearest_distance = world.hit_shadows(r_shadow)
            see_light = distances[world.objects.index(self)] == nearest_distance
            # There is something wrong in the lv

            lv = torch.clamp(u.dot(outward_normal, r_shadow.vel), min=0) # Taking the maximum

            color += self.material.albedo * see_light * lv

        return r, color, intersections

    def intersect(self, r):
        oc = r.pos - self.center
        distances = torch.sqrt(u.dot(oc, oc))
        return distances

    def intersect_straight(self, r, t_max):
        oc = r.pos - self.center
        a = u.dot(r.vel, r.vel) # This will be one always actually
        half_b = u.dot(oc, r.vel)
        c = u.dot(oc, oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        sqrtd = torch.sqrt(torch.maximum(torch.zeros_like(discriminant), discriminant))
        h0 = (-half_b - sqrtd) / a
        h1 = (-half_b + sqrtd) / a

        h = torch.where((h0 > 0.1) & (h0 < h1), h0, h1)
        pred = (discriminant > 0) & (h > 0)
        root = torch.where(pred, h, t_max)
        return root
