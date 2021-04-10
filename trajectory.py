import torch
from libs.sphere import Sphere
from libs.world import World
from libs.camera import Camera
from libs.material import Material
from libs.utils import read_config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import sin, cos, pi
from collections import namedtuple

def make_coor(n):
    phi, theta = np.mgrid[pi/4:pi/2:n, 0:pi:n]
    Coor = namedtuple('Coor', 'r phi theta x y z')
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return Coor(r, phi, theta, x, y, z)

pts=make_coor(15j)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(pts.x,pts.y,pts.z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.scatter(pts.x,pts.y,pts.z)
plt.show()

if __name__ == '__main__':
    config = read_config('config.yml')
