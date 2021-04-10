import torch
from libs.rays import Rays
from libs.image import Image
import libs.utils as u
import matplotlib.pyplot as plt
from math import tan
from tqdm import tqdm


class Camera:
    def __init__(self, config, device='cpu', lookfrom=None):
        if lookfrom is None:
            lookfrom = torch.tensor(eval(config['lookfrom']))
        lookfrom = lookfrom.float()
        lookat = torch.tensor(eval(config['lookat']))
        vup = torch.tensor(eval(config['upvector']))

        aspect_ratio = u.convert_to_float(config['aspect_ratio'])
        theta = u.degrees_to_radians(config['fov'])
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

        self.image_width = config['image_width']
        self.image_height = int(self.image_width // aspect_ratio)
        self.out_shape = (self.image_height, self.image_width, 3)

        self.steps = config['steps']
        self.timestep = config['initial_timestep']
        self.dt_matrix = None

        self.f = u.f_schwarzschild if config['space'] == 'schwarzschild' else u.f_straight
        self.background_color = config['background_color']
        self.antialiasing = config['antialiasing']
        self.evolution = u.verlet if config['evolution_method'] == 'verlet' else u.runge_kutta
        self.debug = config['debug']
        
        self.device = device

    def timestep_init(self, size):
        self.dt_matrix = torch.full(size, self.timestep, device=self.device)

    def render(self, world):
        total_colors = torch.zeros((self.image_width * self.image_height, 3), device=self.device)
        for _ in range(self.antialiasing):
            x = torch.tile(torch.linspace(0, (self.out_shape[1] - 1) / self.out_shape[1], self.out_shape[1]),
                           (self.out_shape[0],)).unsqueeze(1)
            y = torch.repeat_interleave(
                torch.linspace(0, (self.out_shape[0] - 1) / self.out_shape[0], self.out_shape[0]),
                self.out_shape[1]).unsqueeze(1)

            x += torch.rand(x.shape) / self.out_shape[1]
            y += torch.rand(y.shape) / self.out_shape[0]

            ray = Rays(origin=self.origin,
                       directions=self.lower_left_corner + x * self.horizontal + y * self.vertical - self.origin,
                       device=self.device)

            color = torch.full(ray.pos.size(), self.background_color, device=self.device)
            self.timestep_init((self.image_width * self.image_height, 1))

            for _ in tqdm(range(self.steps), disable=not self.debug):
                ray, color = self.step(ray, world, color)

            total_colors += color

        scale = 1 / self.antialiasing
        colors = torch.sqrt(scale * total_colors)
        return Image.from_flat(colors, self.image_width, self.image_height)

    def step(self, r, world, color):
        r.pos, r.vel = self.evolution(r.pos, r.vel, self.f, self.dt_matrix)

        distances, nearest_distance = world.hit(r)

        for i, obj in enumerate(world.objects):
            r, color_obj, intersections = obj.hit(r, distances[i], color, world)
            color = torch.where(intersections, color_obj + color, color)

        self.dt_matrix = u.update_timestep(nearest_distance)
        return r, color
