import torch
from math import pi
from einops import rearrange
import yaml
import matplotlib.pyplot as plt

FARAWAY = 1.0e3
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


def degrees_to_radians(d):
    return d * pi / 180.


def repeat_value_tensor(val, reps, device='cpu'):
    if type(val) != torch.Tensor:
        return torch.tile(torch.tensor(val, device=device), (reps, 1))
    else:
        return torch.tile(val, reps)


def random_in_range(a, b, size):
    '''
    Random float tensor in range [a, b)
    :param a: minimum value
    :param b: maximum value
    :param size: size of the tensor
    :return: tensor
    '''
    return (a - b) * torch.rand(size, device=dev) + b


def random_on_unit_sphere(size):
    # We use the method in https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    # to produce vectors on the surface of a unit sphere

    x = torch.randn(size)
    l = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)).unsqueeze(1)
    x = (x / l).to(dev)

    return x


def unit_sphere(size):
    v = torch.randn(size)
    v = v / v.norm(2, dim=-1, keepdim=True)
    return v.to(dev)


def random_on_unit_sphere_like(t):
    return unit_sphere(t.shape)


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


def f_schwarzschild(r, v):
    c = torch.cross(r, v)
    h2 = dot(c, c)
    return -1.5 * h2 * r / torch.pow(dot(r, r), 2.5)


def f_straight(r, v):
    return 0

def update_timestep(distances, mindis=1.5, maxdis=2, mindt=0.0001, maxdt=1):
    dt = (maxdt - mindt) / (maxdis - mindis) * distances + mindt * maxdis - maxdt * mindis
    dt = torch.clamp(dt, mindt, maxdt)
    return dt