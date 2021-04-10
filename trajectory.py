import torch
from libs.sphere import Sphere
from libs.world import World
from libs.camera import Camera
from libs.material import Material
from libs.utils import read_config
import libs.utils as u
import json

if __name__ == '__main__':
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = read_config('config.yml')

    prev_json_path = u.prepare_dirs(config['run_folder'])
    u.copy_file('config.yml', config['run_folder'])
    prev_json = u.read_previous_json(prev_json_path)
    prev_poses = u.get_previous_poses(prev_json)

    light_pos = eval(config['light_pos'])

    material_back_sphere = Material('lambertian', torch.tensor((230, 26, 11), device=dev))
    material_front_sphere = Material('lambertian', torch.tensor((31, 146, 224), device=dev))

    # World
    world = World(dev)
    world.add(Sphere(torch.tensor((0.0, 0, 6.)), 2, material_back_sphere, device=dev))  # Back sphere
    world.add(Sphere(torch.tensor((0.0, 0, 0.)), 2, material_front_sphere, device=dev))  # Front sphere
    world.add_light(torch.tensor(light_pos, device=dev))

    x_traj, y_traj, z_traj = u.get_cam_trajectory(config)

    for n, (i, j, k) in enumerate(zip(x_traj, y_traj, z_traj)):
        lookfrom = torch.tensor((i, j, k))
        if prev_poses is not None:
            if (lookfrom == prev_poses).all(1).any():
                continue
        lookfrom.to(dev)
        cam = Camera(config, dev)

        with torch.no_grad():
            image = cam.render(world)

        image.save(f'{config["run_folder"]}/imgs/img_{n}.png', flip=True)
        results_dict = {'file_path': f'/imgs/img_{n}.png',
                        'cam_pos': list(lookfrom.detach().cpu().numpy())
                        }
        u.update_json(prev_json_path, results_dict)

