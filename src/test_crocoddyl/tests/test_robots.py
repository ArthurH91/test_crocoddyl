from os.path import dirname, join, abspath
import numpy as np
import matplotlib.pyplot as plt
import hppfcl
import example_robot_data as robex
import pinocchio as pin
import argparse

hppfcl.WITH_OCTOMAP = False

# PARSER 

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--panda", help="Chose the panda to do the unit tests", action="store_true", default=False
)
args = parser.parse_args()

### HELPERS

YELLOW_FULL = np.array([1, 1, 0, 1.0])
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

WITH_DISPLAY = True

PANDA = args.panda

def wrapper_robot():
    """Load the robot from the models folder.

    Returns:
        rmodel, vmodel, cmodel: Robot model, visual model & collision model of the robot.
    """

    ### LOADING THE ROBOT
    pinocchio_model_dir = join(dirname(dirname(dirname(str(abspath(__file__))))), "models")
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

class RobotWrapper:
    def __init__(self, scale=1.0, name_robot="ur10"):
        """Initialize the wrapper with a scaling number of the target and the name of the robot wanted to get unwrapped.

        Parameters
        ----------
        _scale : float, optional
            Scale of the target, by default 1.0
        name_robot : str, optional
            Name of the robot wanted to get unwrapped, by default "ur10"
        """

        self._scale = scale
        self._robot = robex.load(name_robot)
        self._rmodel = self._robot.model
        self._color = np.array([249, 136, 126, 255]) / 255

    def __call__(self, target=False):
        """Create a robot with a new frame at the end effector position and place a hppfcl: ShapeBase cylinder at this position.

        Parameters
        ----------
        target : bool, optional
            Boolean describing whether the user wants a target or not, by default False

        Returns
        -------
        _robot
            Robot description of the said robot
        _rmodel
            Model of the robot
        _gmodel
            Geometrical model of the robot


        """

        # Creation of the frame for the end effector by using the frame tool0, which is at the end effector pose.
        # This frame will be used for the position of the cylinder at the end of the effector.
        # The cylinder is used to have a HPPFCL shape at the end of the robot to make contact with the target

        # Obtaining the frame ID of the frame tool0
        ID_frame_tool0 = self._rmodel.getFrameId("tool0")
        # Obtaining the frame tool0
        frame_tool0 = self._rmodel.frames[ID_frame_tool0]
        # Obtaining the parent joint of the frame tool0
        parent_joint = frame_tool0.parent
        # Obtaining the placement of the frame tool0
        Mf_endeff = frame_tool0.placement

        # Creation of the geometrical model
        self._gmodel = self._robot.visual_model

        # Creation of the cylinder at the end of the end effector

        # Setting up the raddi of the cylinder
        endeff_radii = 1e-2
        # Creating a HPPFCL shape
        endeff_shape = hppfcl.Sphere(endeff_radii)
        # Creating a pin.GeometryObject for the model of the _robot
        geom_endeff = pin.GeometryObject(
            "endeff_geom", ID_frame_tool0, parent_joint, endeff_shape, Mf_endeff
        )
        geom_endeff.meshColor = self._color
        # Add the geometry object to the geometrical model
        self._gmodel.addGeometryObject(geom_endeff)

        if target:
            self._create_target()

        return self._robot, self._rmodel, self._gmodel

    def _create_target(self):
        """Updates the version of the robot models with a sphere that can be used as a target.

        Returns
        -------
        _robot
            Robot description of the said robot
        _rmodel
            Model of the robot
        _gmodel
            Geometrical model of the robot
        """

        # Setup of the shape of the target (a sphere here)
        r_target = 5e-2 * self._scale

        # Creation of the target

        # Creating the frame of the target

        self._M_target = self._generate_reachable_SE3_vector()

        target_frame = pin.Frame(
            "target", 0, self._rmodel.getFrameId("universe"), self._M_target, pin.BODY
        )
        target = self._rmodel.addFrame(target_frame, False)
        T_target = self._rmodel.frames[target].placement
        target_shape = hppfcl.Sphere(r_target)
        geom_target = pin.GeometryObject(
            "target_geom",
            self._rmodel.getFrameId("universe"),
            self._rmodel.getJointId("universe"),
            target_shape,
            T_target,
        )

        geom_target.meshColor = self._color
        self._gmodel.addGeometryObject(geom_target)

    def _generate_reachable_SE3_vector(self):
        """Generate a SE3 vector that can be reached by the robot.

        Returns
        -------
        Reachable_SE3_vector
            SE3 Vector describing a reachable position by the robot
        """

        # Generate a random configuration of the robot, with consideration to its limits
        self._q_target = pin.randomConfiguration(self._rmodel)
        # Creation of a temporary model.Data, to have access to the forward kinematics.
        ndata = self._rmodel.createData()
        # Updating the model.Data with the framesForwardKinematics
        pin.framesForwardKinematics(self._rmodel, ndata, self._q_target)

        return ndata.oMf[self._rmodel.getFrameId("tool0")]
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


def ddist_numdiff(q):
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
        e[i] = 1e-6
        j_diff[i] = (dist(q + e) - dist(q - e)) / e[i] / 2
    return j_diff

def ddist_florent(q):
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
        pin.LOCAL_WORLD_ALIGNED,
    )

    # Computing the distance
    distance = hppfcl.distance(
        shape1.geometry,
        hppfcl.Transform3f(shape1_placement.rotation, shape1_placement.translation),
        shape2.geometry,
        hppfcl.Transform3f(shape2_placement.rotation, shape2_placement.translation),
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


if __name__ == "__main__":
    # Loading the robot
    
    if PANDA:
        rmodel, vmodel, cmodel = wrapper_robot()
    else:
        robot_wrapper = RobotWrapper()
        robot, rmodel, cmodel = robot_wrapper(target=True)
    # Creating the datas model
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Initial configuration
    q0 = pin.neutral(rmodel)

    # Number of joints
    nq = rmodel.nq

    # Updating the models
    pin.framesForwardKinematics(rmodel, rdata, q0)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q0)

    # Creating the shapes for the collision detection.
    
    if PANDA:
        shape1_id = cmodel.getGeometryId("panda2_link7_sc_2")
        shape2_id = cmodel.getGeometryId("panda2_link3_sc_1")

    else: 
        shape1_id = cmodel.getGeometryId("endeff_geom")
        shape2_id = cmodel.getGeometryId("target_geom")

    # Making sure the shape exists
    assert shape1_id <= len(cmodel.geometryObjects)

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
    assert shape2_id <= len(cmodel.geometryObjects)

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


    # Distance & Derivative results from hppfcl
    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()

    # Computing the distance between shape 1 & shape 2 at q0 & comparing with the distance anatically computed
    print(f"dist(q) : {round(dist(q0),6)}")
    dist2 = np.linalg.norm(shape1_placement.translation - shape2_placement.translation)
    print(
        f"np.linalg.norm(dist2) : {dist2 - shape1_geom.radius - shape2_geom.radius:.6f}"
    )

    # Varying k from -pi to pi for a single joint to explore the derivatives of the distance through its full rotation.
    distance_list = []
    ddist_list = []
    ddist_numdiff_list = []
    theta = np.linspace(-np.pi, np.pi, 10000)
    for k in theta:
        if PANDA:
            q = np.array([0, 0, 0, k, 0, 0,0])
        else:
            q = np.array([0, 0, 0, k, 0, 0])

        d = dist(q)
        
        distance_list.append(d)
        ddist_numdiff_val = ddist_numdiff(q)
        ddist_florent_val = ddist_florent(q)
        ddist_numdiff_list.append(ddist_numdiff_val)
        ddist_list.append(ddist_florent_val)

    plots = [331, 332, 333, 334, 335, 336, 337]
    for k in range(nq):
        plt.subplot(plots[k])
        plt.plot(theta, np.array(ddist_list)[:, k], "--", label="florent")
        plt.plot(theta, np.array(ddist_numdiff_list)[:, k], label="numdiff")
        plt.title("joint" + str(k))
        plt.ylabel(f"Distance derivative w.r.t. joint {k}")
        plt.legend()
    plt.suptitle("Numdiff derivatives vs florent derivatives through theta, the rotation angle of the joint 4.")
    plt.show()
