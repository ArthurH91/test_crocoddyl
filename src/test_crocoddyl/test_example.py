import numpy as np
import pinocchio as pin
import hppfcl
import meshcat
import pydiffcol

import meshcat.geometry as g
import meshcat.transformations as tf

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


def create_visualizer(grid=False, axes=False):
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    vis.delete()
    if not grid:
        vis["/Grid"].set_property("visible", False)
    if not axes:
        vis["/Axes"].set_property("visible", False)
    return vis


def select_strategy(strat: str, verbose: bool = False):
    req = hppfcl.DistanceRequest()
    req.gjk_initial_guess = hppfcl.GJKInitialGuess.CachedGuess
    req.gjk_convergence_criterion = hppfcl.GJKConvergenceCriterion.DualityGap
    req.gjk_convergence_criterion_type = hppfcl.GJKConvergenceCriterionType.Absolute
    req.gjk_tolerance = 1e-8
    req.epa_tolerance = 1e-8
    req.epa_max_face_num = 1000
    req.epa_max_vertex_num = 1000
    req.epa_max_iterations = 1000

    req_diff = pydiffcol.DerivativeRequest()
    req_diff.warm_start = np.array([1.0, 0.0, 0.0])
    req_diff.support_hint = np.array([0, 0], dtype=np.int32)

    if strat == "finite_differences":
        req_diff.derivative_type = pydiffcol.DerivativeType.FiniteDifferences
    elif strat == "zero_order_gaussian":
        req_diff.derivative_type = pydiffcol.DerivativeType.ZeroOrderGaussian
    elif strat == "first_order_gaussian":
        req_diff.derivative_type = pydiffcol.DerivativeType.FirstOrderGaussian
    elif strat == "first_order_gumbel":
        req_diff.derivative_type = pydiffcol.DerivativeType.FirstOrderGumbel
    else:
        raise NotImplementedError

    if verbose:
        print("Strategy: ", req_diff.derivative_type)
        print("Noise: ", req_diff.noise)
        print("Num samples: ", req_diff.num_samples)

    return req, req_diff


def get_transform(T_: hppfcl.Transform3f):
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


def renderEllipsoid(
    vis: meshcat.Visualizer,
    radii: np.ndarray,
    e_name: str,
    T_: hppfcl.Transform3f,
    color=np.array([1.0, 1.0, 1.0, 1.0]),
):
    vis[e_name].set_object(g.Ellipsoid(radii), meshcat_material(*color))
    T = get_transform(T_)
    vis[e_name].set_transform(T)


def renderSphere(
    vis: meshcat.Visualizer,
    radii: float,
    e_name: str,
    T_: hppfcl.Transform3f,
    color=np.array([1.0, 1.0, 1.0, 1.0]),
):
    vis[e_name].set_object(g.Sphere(radii), meshcat_material(*color))
    T = get_transform(T_)
    vis[e_name].set_transform(T)


def renderPoint(
    vis: meshcat.Visualizer,
    point: np.ndarray,
    point_name: str,
    color=np.ones(4),
    radius_point=0.001,
):
    hex_color, opacity = rgbToHex(color)
    vis[point_name].set_object(
        g.Sphere(radius_point), g.MeshLambertMaterial(color=hex_color, opacity=opacity)
    )
    vis[point_name].set_transform(tf.translation_matrix(point))


def draw_shape(
    vis: meshcat.Visualizer,
    shape: hppfcl.ShapeBase,
    name: str,
    M: pin.SE3,
    color: np.ndarray,
    render_faces=True,
):
    if isinstance(shape, hppfcl.Ellipsoid):
        renderEllipsoid(vis, shape.radii, name, M, color)
    if isinstance(shape, hppfcl.Sphere):
        renderSphere(vis, shape.radius, name, M, color)


"""     if isinstance(shape, hppfcl.ConvexBase):
        renderConvex(vis, shape, name, M, color, render_faces) """


def render_scene(
    vis: meshcat.visualizer, gmodel: pin.GeometryModel, gdata: pin.GeometryData
):
    "Print in the meshcat server the shapes created in the function create_robot. Call updateGeometryPlacements before this function"
    for gid, geom_object in enumerate(gmodel.geometryObjects):
        shape: hppfcl.ShapeBase = geom_object.geometry
        M = gdata.oMg[gid]
        draw_shape(vis, shape, geom_object.name, M, GREEN)


def create_robot(M_target=pin.SE3.Identity()):
    rmodel: pin.Model = pin.Model()

    # Creation of the joint models variables used to setup the robot

    revolut_joint_y = pin.JointModelFreeFlyer()

    # Id of the universe
    joint_universe = rmodel.getJointId("universe")

    # Creation of the robot

    # Creation of the joint 1 (a revolut one around the y axis)
    Mj1 = pin.SE3.Identity()
    Mj1.translation = np.array([0, 0, 3.5])
    joint1 = rmodel.addJoint(joint_universe, revolut_joint_y, Mj1, "joint1")

    # Creation of the frame F1
    Mf1 = pin.SE3.Identity()
    link1_frame = pin.Frame("link1", joint1, Mf1, pin.BODY)
    link1 = rmodel.addFrame(link1_frame, False)

    # Creation of the frame for the target object
    target_frame = pin.Frame(
        "target", rmodel.getJointId("universe"), M_target, pin.BODY
    )
    target = rmodel.addFrame(target_frame, False)

    # Creation of the geometrical model

    gmodel = pin.GeometryModel()

    # Creation of the shape of the first link
    r1 = 1.5  # Radii of the Ellipsoid

    # Creation of the shape of the 2nd link
    endeff_shape = hppfcl.Sphere(r1)
    endeff_geom = pin.GeometryObject(
        "endeff_geom", joint1, joint1, rmodel.frames[joint1].placement, endeff_shape
    )
    _ = gmodel.addGeometryObject(endeff_geom)

    # Creation of the shape of the target
    T_target = rmodel.frames[target].placement
    target_shape = hppfcl.Sphere(r1)
    target_geom = pin.GeometryObject(
        "target_geom", rmodel.getJointId("universe"), target, T_target, target_shape
    )
    _ = gmodel.addGeometryObject(target_geom)

    return rmodel, gmodel

def dist(q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(
            rmodel,
            rdata,
            gmodel,
            gdata,
            q,
        )
    # Computing the distance
    distance = pydiffcol.distance(
        shape1_geom,
        shape1_placement,
        shape2_geom,
        shape2_placement,
        req,
        res,
    )
    return distance

def dist_numdiff(q):
    j_diff = np.zeros(nq)
    fx = f(data, x[: nq])
    for i in range(nq):
        e = np.zeros(nq)
        e[i] = 1e-6
        j_diff[i] = (f(data, x[: nq] + e) - fx) / e[i]
    _J = j_diff
    
if __name__ == "__main__":
    # pin.seed(0)

    # Setup
    vis = create_visualizer(axes=True)
    target_pose = pin.SE3.Identity()
    target_pose.translation = np.array([0, 10, 0])
    rmodel, gmodel = create_robot(pin.SE3.Random())
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    q = pin.neutral(rmodel)
    nq = rmodel.nq
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)
    req, req_diff = select_strategy("first_order_gaussian")
    res = pydiffcol.DistanceResult()
    res_diff = pydiffcol.DerivativeResult()
    render_scene(vis, gmodel, gdata)

    pin.updateGeometryPlacements(
        rmodel,
        rdata,
        gmodel,
        gdata,
        q,
    )

    shape1_id = gmodel.getGeometryId("endeff_geom")
    shape1 = gmodel.geometryObjects[shape1_id]
    # Getting the geometry of the shape 1
    shape1_geom = shape1.geometry

    # Getting its pose in the world reference
    shape1_placement = gdata.oMg[shape1_id]

    # Doing the same for the second shape.
    shape2_id = gmodel.getGeometryId("target_geom")
    shape2 = gmodel.geometryObjects[shape2_id]
    shape2_geom = shape2.geometry
    shape2_placement = gdata.oMg[shape2_id]

    print(dist(q))
    
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)

    pin.computeJointJacobians(rmodel, rdata, q)

    pydiffcol.distance_derivatives(
    shape1_geom,
    shape1_placement,
    shape2_geom,
    shape2_placement,
    req,
    res,
    req_diff,
    res_diff,
)

    jacobian = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        pin.LOCAL,
    )
    
    print(np.dot(res_diff.ddist_dM1, jacobian))
    print(jacobian)
    print(res_diff.ddist_dM1)
    
    