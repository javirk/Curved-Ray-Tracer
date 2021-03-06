import torch
from math import pi
from einops import rearrange
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import json

FARAWAY = 1.0e3


def dot(a, b):
    '''
    Dot product between two vectors
    :return: torch tensor
    '''
    return torch.einsum('...i,...i', a, b).unsqueeze(-1)


def unit_vector(a, dim=None):
    '''
    Returns the unit vector with the direction of a
    :param dim:
    :param a:
    :return:
    '''
    return a / a.norm(dim=dim).unsqueeze(-1)


def unit_vector_np(a, axis=None):
    return a / np.linalg.norm(a, axis=axis)[..., np.newaxis]


def degrees_to_radians(d):
    if type(d) == str:
        d = eval(d)
    if type(d) == list or type(d) == tuple:
        return [x * pi / 180. for x in d]
    return d * pi / 180.


def repeat_value_tensor(val, reps, device='cpu'):
    if type(val) != torch.Tensor:
        return torch.tile(torch.tensor(val, device=device), (reps, 1))
    else:
        return torch.tile(val, reps)


def random_in_range(a, b, size, device='cpu'):
    '''
    Random float tensor in range [a, b)
    :param a: minimum value
    :param b: maximum value
    :param size: size of the tensor
    :param device:
    :return: tensor
    '''
    return (a - b) * torch.rand(size, device=device) + b


def random_on_unit_sphere(size, device='cpu'):
    # We use the method in https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    # to produce vectors on the surface of a unit sphere

    x = torch.randn(size)
    l = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)).unsqueeze(1)
    x = (x / l).to(device)

    return x


def unit_sphere(size, device='cpu'):
    v = torch.randn(size)
    v = v / v.norm(2, dim=-1, keepdim=True)
    return v.to(device)


def random_on_unit_sphere_like(t, device='cpu'):
    return unit_sphere(t.shape, device)


def plot_t(t, width=400, height=225):
    assert width * height == t.shape[0], 'Dimensions mismatch'
    data = rearrange(t, '(h w) c -> h w c', w=width, h=height)
    plt.imshow(data.cpu())


def plot_dir(r):
    if type(r) != torch.Tensor:
        r = r.directions
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r = r.cpu().numpy()
    ax.scatter(r[:, 0], r[:, 1], r[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def runge_kutta(r, v, f, dt=0.1):
    k1 = dt * v
    l1 = dt * f(r, v)
    k2 = dt * (v + 0.5 * l1)
    l2 = dt * f(r + 0.5 * k2, v)
    k3 = dt * (v + 0.5 * l2)
    l3 = dt * f(r + 0.5 * k3, v)
    k4 = dt * (v + l3)
    l4 = dt * f(r + k4, v)

    r1 = r + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    v1 = v + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)

    return r1, v1


def verlet(r, v, f, dt=0.1):
    v05 = v + 0.5 * f(r, v) * dt
    r1 = r + v05 * dt
    v1 = v05 + 0.5 * f(r1, v) * dt / 2
    return r1, v1


# def f_schwarzschild(r, v, r0):
#     r = r - r0
#     c = torch.cross(r, v)
#     h2 = dot(c, c)
#     return -1.5 * h2 * r / torch.pow(dot(r, r), 2.5)
#
#
# def f_straight(r, v):
#     return 0

class Force:
    def __init__(self, r0, space):
        self.r0 = r0
        self.f = self.f_schwarzschild if space == 'schwarzschild' else self.f_straight

    def __call__(self, r, v):
        return self.f(r, v)

    def f_schwarzschild(self, r, v):
        r = r - self.r0
        c = torch.cross(r, v)
        h2 = dot(c, c)
        return -1.5 * h2 * r / torch.pow(dot(r, r), 2.5)

    def f_straight(self, r, v):
        return 0


def update_timestep(distances, mindis=1.5, maxdis=3, mindt=0.0001, maxdt=1):
    dt = (maxdt - mindt) / (maxdis - mindis) * distances + mindt * maxdis - maxdt * mindis
    dt = torch.clamp(dt, mindt, maxdt)
    return dt


def get_cam_trajectory(config):
    n_points = config['total_points']
    assert np.sqrt(n_points) == int(np.sqrt(n_points)), 'Number of points must be a square number'
    points_per_line = complex(0, np.sqrt(n_points))
    initial_theta, final_theta = degrees_to_radians(config['theta'])
    initial_phi, final_phi = degrees_to_radians(config['phi'])

    phi, theta = np.mgrid[initial_theta:final_theta:points_per_line,
                 initial_phi:final_phi:points_per_line]

    x = config['r'] * np.sin(phi) * np.cos(theta)
    y = config['r'] * np.cos(phi)
    z = - config['r'] * np.sin(phi) * np.sin(theta)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return x, y, z


def prepare_dirs(f, camera_angle=None):
    os.makedirs(f, exist_ok=True)
    os.makedirs(f + '/imgs', exist_ok=True)
    json_dir = f + '/poses.json'
    if not os.path.exists(json_dir):
        assert camera_angle is not None
        data = {'frames': [], 'camera_angle_x': degrees_to_radians(camera_angle)}
        with open(json_dir, "w") as file:
            json.dump(data, file)
    return json_dir


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


def read_previous_json(src):
    try:
        with open(src, "r") as read_file:
            data = json.load(read_file)
    except FileNotFoundError:
        data = None
    return data


def get_previous_poses(data):
    if len(data['frames']) == 0:
        return None
    p = []
    for f in data['frames']:
        p.append(f['transform_matrix'])
    return torch.tensor(p)


def update_json(src, update_data):
    with open(src, "r") as file:
        data = json.load(file)
    data['frames'].append(update_data)
    with open(src, "w") as file:
        json.dump(data, file)


def get_transform_matrix(eye, center, up, final_r=4):
    '''
    This returns the cam to world matrix in blender coordinates.
    :param eye:
    :param center:
    :param up:
    :param final_r:
    :return:
    '''
    # Renormalizing the camera position first
    eye = np.array(eye)
    r = np.linalg.norm(eye, axis=-1)
    eye = eye / r * final_r

    zaxis = unit_vector_np(eye - center)
    xaxis = unit_vector_np(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)

    transform_matrix = np.eye(4)
    transform_matrix[:-1, 0] = xaxis
    transform_matrix[:-1, 1] = yaxis
    transform_matrix[:-1, 2] = zaxis
    transform_matrix[:-1, 3] = eye
    return transform_matrix


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")