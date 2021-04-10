import torch
from libs.rays import Rays
from libs.image import Image
import libs.utils as u
import matplotlib.pyplot as plt
from math import tan
from tqdm import tqdm


class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, image_width, aspect_ratio, space, steps, timestep):
        aspect_ratio = u.convert_to_float(aspect_ratio)
        theta = u.degrees_to_radians(vfov)
        h = tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        w = u.unit_vector(lookfrom - lookat, dim=0)
        s = u.unit_vector(torch.cross(vup, w), dim=0)
        v = torch.cross(w, s)

        self.origin = lookfrom
        self.horizontal = viewport_width * s
        self.vertical = viewport_height * v
        self.lower_left_corner = self.origin - self.horizontal / 2 - self.vertical / 2 - w

        self.image_width = image_width
        self.image_height = int(self.image_width // aspect_ratio)
        self.out_shape = (self.image_height, self.image_width, 3)

        self.steps = steps
        self.timestep = timestep
        self.dt_matrix = None

        self.f = u.f_schwarzschild if space == 'schwarzschild' else u.f_straight

    def timestep_init(self, size):
        self.dt_matrix = torch.full(size, self.timestep, device=u.dev)

    def render(self, world, antialiasing=1, method='runge_kutta'):
        evolution = u.verlet if method == 'verlet' else u.runge_kutta
        total_colors = torch.zeros((self.image_width * self.image_height, 3), device=u.dev)
        for _ in range(antialiasing):
            x = torch.tile(torch.linspace(0, (self.out_shape[1] - 1) / self.out_shape[1], self.out_shape[1]),
                           (self.out_shape[0],)).unsqueeze(1)
            y = torch.repeat_interleave(
                torch.linspace(0, (self.out_shape[0] - 1) / self.out_shape[0], self.out_shape[0]),
                self.out_shape[1]).unsqueeze(1)
            if antialiasing != 1:
                x += torch.rand(x.shape) / self.out_shape[1]
                y += torch.rand(y.shape) / self.out_shape[0]

            ray = Rays(origin=self.origin,
                       directions=self.lower_left_corner + x * self.horizontal + y * self.vertical - self.origin)

            color = torch.ones(ray.pos.size(), device=u.dev) * 0.05
            self.timestep_init((self.image_width * self.image_height, 1))

            for t in tqdm(range(self.steps)):
                ray, color = self.step(ray, world, color, evolution)

            total_colors += color

        scale = 1 / antialiasing
        colors = torch.sqrt(scale * total_colors)
        return Image.from_flat(colors, self.image_width, self.image_height)

    def step(self, r, world, color, evolution):
        r.pos, r.vel = evolution(r.pos, r.vel, self.f, self.dt_matrix)

        distances, nearest_distance = world.hit(r)

        for i, obj in enumerate(world.objects):
            r, color_obj, intersections = obj.hit(r, distances[i], color, world)
            color = torch.where(intersections, color_obj + color, color)

        self.dt_matrix = u.update_timestep(nearest_distance)
        return r, color
