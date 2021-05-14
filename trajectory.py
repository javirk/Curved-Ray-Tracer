import torch
from libs.sphere import Sphere
from libs.world import World
from libs.camera import Camera
from libs.material import Material
from libs.utils import read_config
import libs.utils as u
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
    try:
        eps = eval(config['epsilon'])
    except KeyError:
        eps = 1e-5

    prev_json_path = u.prepare_dirs(config['run_folder'], camera_angle=config['hfov'])
    u.copy_file(config_path, config['run_folder'])
    prev_json = u.read_previous_json(prev_json_path)
    prev_poses = u.get_previous_poses(prev_json)

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

    r0 = world.objects[1].center

    if config['trajectory'] == 'circular':
        x_traj, y_traj, z_traj = u.get_cam_trajectory(config)

        for n, (i, j, k) in enumerate(zip(x_traj, y_traj, z_traj)):
            lookfrom = torch.tensor((i, j, k))
            if prev_poses is not None:
                if (torch.abs(lookfrom - prev_poses) < eps).all(1).any():
                    print(str(n) + ' is already done')
                    continue

            print('Generating image ' + str(n))
            lookfrom.to(dev)
            cam = Camera(config, r0, lookfrom=lookfrom, device=dev)

            with torch.no_grad():
                image = cam.render(world)

            image.save(f'{config["run_folder"]}/imgs/img_{n}.png', flip=True)
            results_dict = {'file_path': f'/imgs/img_{n}.png',
                            'transform_matrix': u.listify_matrix(cam.matrix_world)
                            }
            u.update_json(prev_json_path, results_dict)

    elif config['trajectory'] == 'random':
        if config['overwrite']:
            it = range(0, config['total_points'])
        else:
            assert prev_poses.shape[0] < config['total_points'], 'The folder is already full.'
            it = range(prev_poses.shape[0], config['total_points'])

        for n in it:
            lookfrom = u.random_on_unit_sphere((1, 3)) * config['r']

            print('Generating image ' + str(n))
            cam = Camera(config, r0, lookfrom=lookfrom.squeeze(), device=dev)

            with torch.no_grad():
                image = cam.render(world)

            image.save(f'{config["run_folder"]}/imgs/img_{n}.png', flip=True)
            results_dict = {'file_path': f'/imgs/img_{n}.png',
                            'transform_matrix': u.listify_matrix(cam.matrix_world)
                            }
            u.update_json(prev_json_path, results_dict)


