import torch
from libs.sphere import Sphere
from libs.world import World
from libs.camera import Camera
from libs.material import Material
from libs.utils import read_config
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config-path',
                    default='config.yml',
                    type=str)

FLAGS, unparsed = parser.parse_known_args()
config_path = FLAGS.config_path

if __name__ == '__main__':
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = read_config(config_path)

    light1_pos = eval(config['light1_pos'])
    light2_pos = eval(config['light2_pos'])

    material_red = Material('lambertian', torch.tensor((0.8, 0.023, 0.011), device=dev))
    material_blue = Material('lambertian', torch.tensor((0.004, 0.023, 0.8), device=dev))

    # World
    world = World(dev)
    world.add(Sphere(torch.tensor((0., 0., 3)), 2, material_blue, device=dev))  # Front sphere
    world.add(Sphere(torch.tensor((0., 0., -3)), 2, material_red, device=dev))  # Back sphere
    world.add_light(torch.tensor(light1_pos, device=dev))
    world.add_light(torch.tensor(light2_pos, device=dev))

    r0 = torch.tensor((0., 0., -3.), device=dev)

    cam = Camera(config, r0, dev)

    with torch.no_grad():
        image = cam.render(world)

    # image.show(flip=True)
    image.save('render.png', flip=True)
