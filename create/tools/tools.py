
"""
    给粒子加自转角速度等函数
"""
import numpy as np


def add_omg(x: np.array, y: np.array, z: np.array, omg: list, center: list = None) -> [np.array, np.array, np.array]:
    """
        计算每个粒子由于全局的角速度增加的速度
    :param x: 粒子横坐标
    :param y: 粒子y坐标
    :param w: 全局角速度
    :param center: 瞬时旋转中心
    :return: 粒子增加的x、y方向速度
    """
    if center is None:
        center = [np.mean(x), np.mean(y), np.mean(z)]

    dx = x - center[0]
    dy = y - center[1]
    dz = z - center[2]

    u = 0
    v = dz * omg[0]
    w = -dy * omg[0]

    u += -dz * omg[1]
    w += dx * omg[1]

    u += dy * omg[2]
    v += -dx * omg[2]

    return u, v, w


def get_velocity(V, direction):
    direction = np.array(direction) / max(np.sum(np.array(direction) ** 2) ** 0.5, 1e-10)
    v = V * direction
    return v[0], v[1], v[2]


def gravity(x, y, z, m, x0, y0, z0):
    """
        计算某点引力场
    :param x:  所有粒子横坐标
    :param y:  所有粒子纵坐标
    :param m:  所有粒子质量
    :param x0: 所求引力场点横坐标
    :param y0: 所求引力场点纵坐标
    :return: 引力场两个分量
    """
    G = 6.67e-11
    dx = x - x0
    dy = y - y0
    dz = z - z0
    r = (dx ** 2 + dy ** 2 + dz) ** 0.5

    F = G * m / r ** 3
    return F * dx, F * dy, F * dz


def get_random_rubble_pile(dx, max_radius, center=None, rubble_radius_range=None, rubble_num=None):
    """
        生成碎石堆小行星
    :param dx:                    粒子间隔
    :param max_radius:            碎石堆外径
    :param center:                碎石堆中心
    :param rubble_radius_range:   每块石头半径范围
    :param rubble_num:            碎石数量
    :return:
    """
    from pysph.tools import geometry as G
    import random

    if center is None:
        center = [0., 0., 0.]

    if rubble_radius_range is None:
        rubble_radius_range = [min(0.3 * max_radius, 10.), min(0.5 * max_radius, 40.)]

    if rubble_num is None:
        rubble_num = max(random.randint(10, 100), (max_radius // rubble_radius_range[1]) ** 3 * 4.)
        rubble_num = int(rubble_num)

    x, y, z = G.get_3d_sphere(dx, max_radius + rubble_radius_range[1])

    is_exist = np.full([x.shape[0]], 0.)

    for num in range(rubble_num):
        center_ = np.clip(np.random.randn(3), -2.2, 2.2) / 2.2 * max_radius
        radius = np.clip(np.random.randn(1), -1, 1) * 0.5 * (rubble_radius_range[1] - rubble_radius_range[0]) +\
                 0.5 * (rubble_radius_range[1] + rubble_radius_range[0])
        can_exist = (((x - center_[0]) ** 2 + (y - center_[1]) ** 2 + (z - center_[2]) ** 2)) <= (radius ** 2)
        is_exist = is_exist + can_exist

    is_exist = np.nonzero(is_exist)

    x = x[is_exist]
    y = y[is_exist]
    z = z[is_exist]

    x += center[0]
    y += center[1]
    z += center[2]

    # import matplotlib.pyplot as plt
    # plt.xlim([-1.7 * max_radius, 1.7 * max_radius])
    # plt.ylim([-1.7 * max_radius, 1.7 * max_radius])
    # plt.scatter(x, y)
    # plt.show()

    return x, y, z


def get_ellipsoid(a, b, c, dx, direction=None, center=None):
    from pysph.tools import geometry as G

    if direction is None:
        direction = [1., 0., 0.]

    direction = np.array(direction)
    direction = direction / np.sum(direction ** 2) ** 0.5

    if center is None:
        center = [0., 0., 0.]

    x, y, z = G.get_3d_sphere(dx, max(a, b, c))

    is_exist = (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1.
    is_exist = np.nonzero(is_exist)

    x = x[is_exist]
    y = y[is_exist]
    z = z[is_exist]

    e = np.array([1., 0., 0.]).reshape([-1, 1])
    direction = direction.reshape([-1, 1])

    alpha = np.arccos(
                (e[[1, 2]].T @ direction[[1, 2]] + 1e-10) /
                (np.sum(e[[1, 2]] ** 2) ** 0.5 * np.sum(direction[[1, 2]] ** 2) ** 0.5 + 1e-10)
            )

    beta = np.arccos(
                (e[[0, 2]].T @ direction[[0, 2]] + 1e-10) /
                (np.sum(e[[0, 2]] ** 2) ** 0.5 * np.sum(direction[[0, 2]] ** 2) ** 0.5 + 1e-10)
            )
    gamma = np.arccos(
                (e[[0, 1]].T @ direction[[0, 1]] + 1e-10) /
                (np.sum(e[[0, 1]] ** 2) ** 0.5 * np.sum(direction[[0, 1]] ** 2) ** 0.5 + 1e-10)
            )

    # print(alpha / np.pi, beta / np.pi, gamma / np.pi)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    w = Rz.dot(Ry.dot(Rx))

    array = w @ np.concatenate([x.reshape([1, -1]), y.reshape([1, -1]), z.reshape([1, -1])], 0)

    x, y, z = array

    x += center[0]
    y += center[1]
    z += center[2]

    # import matplotlib.pyplot as plt
    # plt.xlim([-1.7 * max(a, b, c), 1.7 * max(a, b, c)])
    # plt.ylim([-1.7 * max(a, b, c), 1.7 * max(a, b, c)])
    # plt.scatter(x, y)
    # plt.show()

    return x, y, z


def demo():
    import matplotlib.pyplot as plt
    x, y, z = get_random_rubble_pile(2., 100.)
    plt.xlim([-1.7 * 100., 1.7 * 100])
    plt.ylim([-1.7 * 100, 1.7 * 100])
    plt.scatter(x, y)
    plt.show()

    x, y, z = get_ellipsoid(100, 30, 50, 2., direction=[1, 3, 2])
    plt.xlim([-1.7 * 100., 1.7 * 100])
    plt.ylim([-1.7 * 100, 1.7 * 100])
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    demo()

















