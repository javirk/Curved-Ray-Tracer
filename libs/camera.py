import torch
from libs.rays import Rays
from libs.image import Image
import libs.utils as u
import matplotlib.pyplot as plt
from math import tan
from tqdm import tqdm


class Camera:
    def __init__(self, config, r0, device='cpu', lookfrom=None):
        if lookfrom is None:
            lookfrom = torch.tensor(eval(config['lookfrom']))
        lookfrom = lookfrom.float()
        lookat = torch.tensor(eval(config['lookat']))
        vup = torch.tensor(eval(config['upvector']))

        aspect_ratio = u.convert_to_float(config['aspect_ratio'])
        theta = u.degrees_to_radians(config['hfov'])
        w = tan(theta / 2)
        viewport_width = 2.0 * w # * config['focal']
        viewport_height = aspect_ratio * viewport_width

        w = u.unit_vector(lookfrom - lookat, dim=0) # This is a different w. The other one is not used anymore
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

        self.force = u.Force(r0, config['space'])

        self.background_color = config['background_color']
        self.antialiasing = config['antialiasing']
        self.evolution = u.verlet if config['evolution_method'] == 'verlet' else u.runge_kutta
        self.debug = config['debug']
        
        self.device = device
        self.matrix_world = u.get_transform_matrix(lookfrom.cpu().numpy(), lookat.cpu().numpy(), vup.cpu().numpy())

    def timestep_init(self, size):
        self.dt_matrix = torch.full(size, self.timestep, device=self.device)

    def render(self, world, use_alpha=True):
        total_colors = torch.zeros((self.image_width * self.image_height, 3), device=self.device)
        total_alpha = torch.zeros((self.image_width * self.image_height, 1), device=self.device)
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
            alpha = torch.zeros((self.image_width * self.image_height, 1), device=self.device)

            self.timestep_init((self.image_width * self.image_height, 1))

            for _ in tqdm(range(self.steps), disable=not self.debug):
                ray, color, alpha = self.step(ray, world, color, alpha)

            total_colors += color
            total_alpha = torch.logical_or(total_alpha, alpha)

        scale = 1 / self.antialiasing
        total_colors = torch.sqrt(scale * total_colors)
        return Image.from_flat(total_colors, total_alpha, self.image_width, self.image_height, use_alpha=use_alpha)

    def step(self, r, world, color, alpha):
        r.pos, r.vel = self.evolution(r.pos, r.vel, self.force, self.dt_matrix)

        distances, nearest_distance = world.hit(r)

        for i, obj in enumerate(world.objects):
            r, color_obj, intersections = obj.hit(r, distances[i], color, world)
            color = torch.where(intersections, color_obj + color, color)
            alpha = torch.logical_or(intersections, alpha)
            # depth = torch.where(intersections, depth - 1, depth)

        self.dt_matrix = u.update_timestep(nearest_distance)
        return r, color, alpha
