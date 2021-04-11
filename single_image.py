import torch
from libs.sphere import Sphere
from libs.world import World
from libs.camera import Camera
from libs.material import Material
from libs.utils import read_config
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = read_config('config.yml')

    light1_pos = eval(config['light1_pos'])
    light2_pos = eval(config['light2_pos'])

    material_back_sphere = Material('lambertian', torch.tensor((230, 26, 11), device=dev))
    material_front_sphere = Material('lambertian', torch.tensor((31, 146, 224), device=dev))

    # World
    world = World(dev)
    world.add(Sphere(torch.tensor((0.0, 0, 6.)), 2, material_back_sphere, device=dev)) # Back sphere
    world.add(Sphere(torch.tensor((0.0, 0, 0.)), 2, material_front_sphere, device=dev)) # Front sphere
    world.add_light(torch.tensor(light1_pos, device=dev))
    world.add_light(torch.tensor(light2_pos, device=dev))

    cam = Camera(config, dev)

    with torch.no_grad():
        image = cam.render(world)

    # image.show(flip=True)
    image.save('render.png', flip=True)
