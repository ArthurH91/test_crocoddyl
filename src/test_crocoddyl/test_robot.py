from os.path import dirname, join, abspath
import numpy as np

import pinocchio as pin
import hppfcl
import pydiffcol

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper

from utils import BLUE, YELLOW_FULL, select_strategy

def dist(q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(
        rmodel,
        rdata,
        cmodel,
        cdata,
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
    fx = dist(q)
    for i in range(nq):
        e = np.zeros(nq)
        e[i] = 1e-6
        j_diff[i] = (dist(q + e) - fx) / e[i]
    return j_diff

def load_robot():
        ### LOADING THE ROBOT
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)
    srdf_model_path = model_path + "/panda/demo.srdf"

    # Creating the robot
    robot_wrapper = RobotWrapper(
        urdf_model_path=urdf_model_path, mesh_dir=mesh_dir, srdf_model_path=srdf_model_path
    )
    rmodel, cmodel, vmodel = robot_wrapper()
    
    ### CREATING THE TARGET
    TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 1.55]))
    TARGET_POSE.translation = np.array([0, -0.4, 1.5])

    return rmodel, cmodel, vmodel, TARGET_POSE

if __name__ == "__main__":

    rmodel, cmodel, vmodel, TARGET_POSE = load_robot()
    rdata = rmodel.createData()
    cdata = cmodel.createData()


    ### CREATING THE OBSTACLE
    OBSTACLE_RADIUS = 1.5e-1
    OBSTACLE_POSE = pin.SE3.Identity()
    OBSTACLE_POSE.translation = np.array([0.25, -0.45, 1.5])
    OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
        OBSTACLE,
        OBSTACLE_POSE,
    )
    OBSTACLE_GEOM_OBJECT.meshColor = BLUE

    IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

    shape1_id = cmodel.getGeometryId("panda2_link5_sc_4")

    shape1 = cmodel.geometryObjects[shape1_id]
    # Getting the geometry of the shape 1
    shape1_geom = shape1.geometry
    shape1_geom.meshColor = YELLOW_FULL

    ### INITIAL CONFIG OF THE ROBOT
    INITIAL_CONFIG = pin.neutral(rmodel)

    cdata = cmodel.createData()

    ### INITIAL X0
    q = INITIAL_CONFIG


    nq = rmodel.nq
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    req, req_diff = select_strategy("first_order_gaussian")
    res = pydiffcol.DistanceResult()
    res_diff = pydiffcol.DerivativeResult()

    pin.updateGeometryPlacements(
        rmodel,
        rdata,
        cmodel,
        cdata,
        q,
    )
    
    shape1_id = cmodel.getGeometryId("panda2_link5_sc_4")
    shape1 = cmodel.geometryObjects[shape1_id]
    # Getting the geometry of the shape 1
    shape1_geom = shape1.geometry

    # Getting its pose in the world reference
    shape1_placement = cdata.oMg[shape1_id]

    # Doing the same for the second shape.
    shape2_id = cmodel.getGeometryId("obstacle")
    shape2 = cmodel.geometryObjects[shape2_id]
    shape2_geom = shape2.geometry
    shape2_placement = cdata.oMg[shape2_id]


    print(f"dist(q) : {round(dist(q),6)}")
    dist2 = np.linalg.norm(shape1_placement.translation - shape2_placement.translation)
    print(f"np.linalg.norm(dist2) : {round(np.linalg.norm(dist2) - OBSTACLE_RADIUS - shape1_geom.radius,6)}")
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

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
    print(f"res diff ddist_dm1 : {res_diff.ddist_dM1}")
    print(f"jacobian : {jacobian}")
    print(f"np.dot(res_diff.ddist_dM1, jacobian) : {np.dot(res_diff.ddist_dM1, jacobian)}")
    print(f"dist_numdiff(q) : {dist_numdiff(q)}")
    print(f"diff fdiff : {dist_numdiff(q) - np.dot(res_diff.ddist_dM1, jacobian) }")
    assert np.isclose(
        np.linalg.norm(np.dot(res_diff.ddist_dM1, jacobian)),
        np.linalg.norm(dist_numdiff(q)),
    )

    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis, meshcatVis = MeshcatVis.visualize(
        TARGET_POSE,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
    )
    vis.display(q)