from os.path import dirname, join, abspath
import numpy as np
import matplotlib.pyplot as plt
import hppfcl

import pinocchio as pin


### HELPERS

YELLOW_FULL = np.array([1, 1, 0, 1.0])
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

WITH_DISPLAY = False

def wrapper_robot():
    """Load the robot from the models folder.

    Returns:
        rmodel, vmodel, cmodel: Robot model, visual model & collision model of the robot.
    """
    ### LOADING THE ROBOT
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)
    srdf_model_path = model_path + "/panda/demo.srdf"

    rmodel, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    )

    q0 = pin.neutral(rmodel)
    jointsToLockIDs = [1, 9, 10]

    geom_models = [visual_model, collision_model]
    model_reduced, geometric_models_reduced = pin.buildReducedModel(
        rmodel,
        list_of_geom_models=geom_models,
        list_of_joints_to_lock=jointsToLockIDs,
        reference_configuration=q0,
    )

    visual_model_reduced, collision_model_reduced = (
        geometric_models_reduced[0],
        geometric_models_reduced[1],
    )

    return model_reduced, visual_model_reduced, collision_model_reduced


######################################## DISTANCE & ITS DERIVATIVES COMPUTATION #######################################


def dist(q):
    """Computes the distance with diffcol

    Args:
        q (np.ndarray): Configuration of the robot

    Returns:
        distance : distance between shape 1 & shape 2
    """
    # Computing the distance
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    shape1_placement = cdata.oMg[shape1_id]
    shape2_placement = cdata.oMg[shape2_id]

    distance = hppfcl.distance(
        shape1_geom,
        shape1_placement,
        shape2_geom,
        shape2_placement,
        req,
        res,
    )
    return distance


def dist_numdiff(q):
    """Finite derivative of the dist function at q.

    Args:
        q (np.ndarray): Configuration of the robot

    Returns:
        distance derivative: distance derivative between shape 1 & shape 2
    """
    j_diff = np.zeros(nq)
    fx = dist(q)
    for i in range(nq):
        e = np.zeros(nq)
        e[i] = 1e-12
        j_diff[i] = (dist(q + e) - dist(q)) / e[i]
    return j_diff


def derivative_distance_sphere_sphere(FRAME=pin.LOCAL):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.computeJointJacobians(rmodel, rdata, q)
    pin.updateGeometryPlacements(
        rmodel,
        rdata,
        cmodel,
        cdata,
        q,
    )
    jacobian = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        FRAME,
    )

    # Computing the distance
    distance = hppfcl.distance(
        shape1_geom,
        shape1_placement,
        shape2_geom,
        shape2_placement,
        req,
        res,
    )

    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()

    deriv = (cp1 - cp2).T / np.linalg.norm(cp1 - cp2) @ jacobian[:3]

    return deriv


def derivative_distance_sphere_sphere3(FRAME=pin.LOCAL):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.computeJointJacobians(rmodel, rdata, q)
    pin.updateGeometryPlacements(
        rmodel,
        rdata,
        cmodel,
        cdata,
        q,
    )
    jacobian = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        FRAME,
    )

    # Computing the distance
    distance = hppfcl.distance(
        shape1_geom,
        shape1_placement,
        shape2_geom,
        shape2_placement,
        req,
        res,
    )

    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()

    CP1_SE3 = pin.SE3.Identity()
    CP1_SE3.translation = cp1

    CP2_SE3 = pin.SE3.Identity()
    CP2_SE3.translation = cp2

    deriv = (cp1 - cp2).T / distance @ jacobian[:3]
    return deriv


def dist_numdiff(q):
    """Finite derivative of the dist function.

    Args:
        q (np.ndarray): Configuration of the robot

    Returns:
        distance derivative: distance derivative between shape 1 & shape 2
    """
    j_diff = np.zeros(nq)
    fx = dist(q)
    for i in range(nq):
        e = np.zeros(nq)
        e[i] = 1e-5
        j_diff[i] = (dist(q + e) - dist(q)) / e[i]
    return j_diff


if __name__ == "__main__":
    # Loading the robot
    rmodel, vmodel, cmodel = wrapper_robot()
    # target = pin.SE3.Identity()
    # target.translation = np.array([1,1,1])
    # rmodel, cmodel = create_robot(target)
    # Creating the datas model
    
        
    ### CREATING THE OBSTACLE
    OBSTACLE_RADIUS = 1.5e-1
    OBSTACLE_POSE = pin.SE3.Identity()
    OBSTACLE_POSE.translation = np.array([0.25, -0.625, 1.5])
    # OBSTACLE = hppfcl.Capsule(OBSTACLE_RADIUS, OBSTACLE_RADIUS)
    OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
        OBSTACLE,
        OBSTACLE_POSE,
    )

    IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

    
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Initial configuration
    q0 = pin.neutral(rmodel)
    q = np.array([1,1,1,1,1,1,1])
    # q = pin.neutral(rmodel)

    # Number of joints
    nq = rmodel.nq

    # Updating the models
    pin.framesForwardKinematics(rmodel, rdata, q0)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q0)

    # Creating the shapes for the collision detection.
    shape1_id = cmodel.getGeometryId("panda2_link5_sc_4")

    # Making sure the shape exists
    assert shape1_id <= len(cmodel.geometryObjects) -1 

    # Coloring the sphere
    shape1 = cmodel.geometryObjects[shape1_id]
    shape1.meshColor = BLUE_FULL

    # Getting the geometry of the shape 1
    shape1_geom = shape1.geometry

    # Making sure the shape is a sphere
    assert isinstance(shape1_geom, hppfcl.Sphere)

    # Getting its pose in the world reference
    shape1_placement = cdata.oMg[shape1_id]

    # Doing the same for the second shape.
    shape2_id = cmodel.getGeometryId("obstacle") 
    assert shape2_id <= len(cmodel.geometryObjects) -1

    # Coloring the sphere
    shape2 = cmodel.geometryObjects[shape2_id]
    shape2.meshColor = YELLOW_FULL

    shape2_geom = shape2.geometry
    assert isinstance(shape2_geom, hppfcl.Sphere)

    # Getting its pose in the world reference
    shape2_placement = cdata.oMg[shape2_id]

    if WITH_DISPLAY:
        from wrapper_meshcat import MeshcatWrapper

        # Generating the meshcat visualizer
        MeshcatVis = MeshcatWrapper()
        vis, meshcatVis = MeshcatVis.visualize(
            robot_model=rmodel,
            robot_collision_model=cmodel,
            robot_visual_model=cmodel,
        )
        # Displaying the initial
        vis.display(q0)

    # Distance & Derivative results from diffcol
    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()

    # Computing the distance between shape 1 & shape 2 at q0 & comparing with the distance anatically computed
    print(f"dist(q) : {round(dist(q0),6)}")
    dist2 = np.linalg.norm(shape1_placement.translation - shape2_placement.translation)
    print(
        f"np.linalg.norm(dist2) : {dist2 - shape1_geom.radius - shape2_geom.radius:.6f}"
    )

    deriv = derivative_distance_sphere_sphere()
    deriv3 = derivative_distance_sphere_sphere3()
    deriv_numdiff = dist_numdiff(q)
    np.set_printoptions(precision=3)
    print(f"deriv : {deriv}")
    print(f"deriv3 : {deriv3}")
    print(f"numdif : {deriv_numdiff}")

    # Varying k from -pi to pi for a single joint to explore the derivatives of the distance through its full rotation.
    distance_list = []
    ddist_numdiff_list_local = []
    ddist_list_florent_local = []
    ddist_list_analytical_local = []

    ddist_numdiff_list_lwa = []
    ddist_list_florent_lwa = []
    ddist_list_analytical_lwa = []

    theta = np.linspace(-np.pi, np.pi, 1000)
    for k in theta:
        q = np.array([0,0,0, k,0,0,0])
        d = dist(q)
        FRAME = pin.LOCAL
        distance_list.append(d)
        ddist_numdiff = dist_numdiff(q)
        ddist_ana = derivative_distance_sphere_sphere3(FRAME)
        ddist_florent = derivative_distance_sphere_sphere(FRAME)
        ddist_numdiff_list_local.append(ddist_numdiff)
        ddist_list_analytical_local.append(ddist_ana)
        ddist_list_florent_local.append(ddist_florent)

        FRAME = pin.LOCAL_WORLD_ALIGNED
        ddist_numdiff = dist_numdiff(q)
        ddist_ana = derivative_distance_sphere_sphere3(FRAME)
        ddist_florent = derivative_distance_sphere_sphere(FRAME)
        ddist_numdiff_list_lwa.append(ddist_numdiff)
        ddist_list_analytical_lwa.append(ddist_ana)
        ddist_list_florent_lwa.append(ddist_florent)

        if WITH_DISPLAY:
            vis.display(q)

    plots = [331, 332, 333, 334, 335, 336, 337]
    plt.figure()

    for k in range(nq):
        plt.subplot(plots[k])
        plt.plot(theta, np.array(ddist_numdiff_list_local)[:, k], "--", label="numdiff")
        plt.plot(
            theta,
            np.array(ddist_list_florent_local)[:, k],
            label="florent_local",
            marker=".",
            markersize=5,
        )
        plt.plot(
            theta,
            np.array(ddist_list_analytical_local)[:, k],
            "--",
            label="analytical_local",
        )
        plt.title("joint" + str(k))
        plt.ylabel(f"Distance derivative w.r.t. joint {k}")
    plt.legend()
    plt.suptitle(
        f"pin.LOCAL"
    )
    plt.figure()

    for k in range(nq):
        plt.subplot(plots[k])
        plt.plot(theta, np.array(ddist_numdiff_list_local)[:, k], "--", label="numdiff")
        plt.plot(
            theta,
            np.array(ddist_list_florent_lwa)[:, k],
            label="florent_lwa",
            marker=".",
            markersize=5,
        )
        plt.plot(
            theta,
            np.array(ddist_list_analytical_lwa)[:, k],
            "--",
            label="analytical_lwa",
        )

        plt.title("joint" + str(k))
        plt.ylabel(f"Distance derivative w.r.t. joint {k}")
    plt.legend()

    plt.suptitle(
        f"LWA"    )

    plt.show()