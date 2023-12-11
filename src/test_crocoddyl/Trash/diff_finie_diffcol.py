import numpy as np
import copy

import hppfcl
import pinocchio as pin
import meshcat
import pydiffcol

import meshcat.geometry as g
import meshcat.transformations as tf

from utils import select_strategy
### COLORS

RED = np.array([249, 136, 126, 125]) / 255
RED_FULL = np.array([249, 136, 126, 255]) / 255

GREEN = np.array([170, 236, 149, 125]) / 255
GREEN_FULL = np.array([170, 236, 149, 255]) / 255

BLUE = np.array([144, 169, 183, 125]) / 255
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

YELLOW = np.array([1, 1, 0, 0.5])
YELLOW_FULL = np.array([1, 1, 0, 1.0])

BLACK = np.array([0, 0, 0, 0.5])
BLACK_FULL = np.array([0, 0, 0, 1.0])


def rgbToHex(color):
    if len(color) == 4:
        c = color[:3]
        opacity = color[3]
    else:
        c = color
        opacity = 1.0
    hex_color = "0x%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
    return hex_color, opacity


def meshcat_material(r, g, b, a):
    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256**2 + int(g * 255) * 256 + int(b * 255)
    material.opacity = a
    return material


### HELPERS


def get_transform(T_: hppfcl.Transform3f):
    """Pin.SE3 to transform for meshcat

    Parameters
    ----------
    T_ : hppfcl.Transform3f
        SE3 Transform

    Returns
    -------
    T : np.ndarray
        Transform usable by meshcat

    Raises
    ------
    NotADirectoryError
        _description_
    """
    T = np.eye(4)
    if isinstance(T_, hppfcl.Transform3f):
        T[:3, :3] = T_.getRotation()
        T[:3, 3] = T_.getTranslation()
    elif isinstance(T_, pin.SE3):
        T[:3, :3] = T_.rotation
        T[:3, 3] = T_.translation
    else:
        raise NotADirectoryError
    return T


def create_visualizer():
    """Creation of an empty visualizer.

    Returns
    -------
    vis : Meshcat.Visualizer
        visualizer from meshcat
    """
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viewer.delete()
    return viewer


def numdiff_se3(f, x, eps=1e-8):
    """Estimate df/dx at x with finite diff of step eps

    Parameters
    ----------
    f : function handle
        Function evaluated for the finite differente of its gradient.
    x : np.ndarray
        Array at which the finite difference is calculated
    eps : float, optional
        Finite difference step, by default 1e-6

    Returns
    -------
    jacobian : np.ndarray
        Finite difference of the function f at x.
    """
    
    xc = np.copy(x)
    xc_log = pin.log(xc)
    f0 = np.copy(f(x))
    res = []
    for i in range(len(x)):
        xc_log[i] += eps
        xc_exp = pin.exp(xc_log)
        res.append(copy.copy(f(xc) - f0) / eps)
        xc[i] = x[i]
    return np.array(res).T

def dist(T):
    
    pydiffcol.distance(
        shape1,
        T,
        shape2,
        T2,
        req,
        res,
    )
    
    return(res.w)


# Building the meshcat materials
red = meshcat_material(RED[0], RED[1], RED[2], RED[3])
green = meshcat_material(GREEN[0], GREEN[1], GREEN[2], GREEN[3])
yellow = meshcat_material(YELLOW[0], YELLOW[1], YELLOW[2], YELLOW[3])
blue = meshcat_material(BLUE[0], BLUE[1], BLUE[2], BLUE[3])


if __name__ == "__main__":
    vis = create_visualizer()

    # Shapes
    r1, l1 = 0.5, 1
    shape1 = hppfcl.Capsule(
        r1, l1
    )  # Radius then height, opposite to meshcat.geometry bellow
    r2 = (1.5,1.3,0.7)
    shape2 = hppfcl.Box(r2[0],r2[1], r2[2])

    T1 = pin.SE3.Identity()

    vis["capsule"].set_object(g.Cylinder(l1, r1), red)
    # T2 = pin.SE3.Random()
    T2 = pin.SE3(
        np.array(
            [
                [0.49375832, 0.8400332, -0.2248265, -0.85810032],
                [-0.80183509, 0.33974407, -0.49156327, -0.37152274],
                [-0.336546, 0.42298723, 0.84131956, -0.29580275],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    vis["sphere"].set_object(g.Box(r2), blue)

    vis["sphere"].set_transform(get_transform(T2))

    req, req_diff = select_strategy("first_order_gaussian")
    res = pydiffcol.DistanceResult()
    res_diff = pydiffcol.DerivativeResult()
    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()

    dist1 = pydiffcol.distance(
        shape1,
        T1,
        shape2,
        T2,
        req,
        res,
    )

    # Vis contact points

    r_w = 0.1

    vis["cp1"].set_object(g.Sphere(r_w), green)
    vis["cp1"].set_transform(tf.translation_matrix(cp1))

    vis["cp2"].set_object(g.Sphere(r_w), yellow)
    vis["cp2"].set_transform(tf.translation_matrix(cp2))

    print(dist1)
    print(res.min_distance)


    pydiffcol.distance_derivatives(
        shape1,
        T1,
        shape2,
        T2,
        req,
        res,
        req_diff,
        res_diff,
)


    