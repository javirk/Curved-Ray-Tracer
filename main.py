import torch
from libs.sphere import Sphere
from libs.world import World
from libs.camera import Camera
from libs.material import Material
from libs.utils import read_config
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = read_config('config.yml')

    material_back_sphere = Material('lambertian', torch.tensor((236, 64, 52), device=dev))
    material_front_sphere = Material('lambertian', torch.tensor((52, 161, 235), device=dev))

    # World
    world = World()
    world.add(Sphere(torch.tensor((0.0, 0, 6.)), 2, material_back_sphere)) # Back sphere
    world.add(Sphere(torch.tensor((0.0, 0, 0.)), 2, material_front_sphere)) # Front sphere
    world.add_light(torch.tensor((0., 0, -6.), device=dev))

    lookfrom = torch.tensor((0, 0., -6.))
    lookat = torch.tensor((0., 0., 0.))
    vup = torch.tensor((0., 1., 0.))

    # TODO: This is the most horrible thing I have ever seen
    cam = Camera(lookfrom, lookat, vup, config['fov'], config['image_width'], config['aspect_ratio'],
                 config['space'], config['steps'], config['timestep'], config['background_color'])

    with torch.no_grad():
        image = cam.render(world, antialiasing=config['antialiasing'], method=config['evolution_method'])

    # image.show(flip=True)
    image.save('test_1.png', flip=True)
