from os.path import dirname, join, abspath
import argparse
import time

import numpy as np
import pinocchio as pin
import hppfcl

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from ocp_panda_reaching_col import OCPPandaReachingCol

from utils import get_transform


###* PARSERS
parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--display", help="display the results", action="store_true", default=False
)
parser.add_argument(
    "-maxit",
    "--maxit",
    help="number max of iterations of the solver",
    default=250,
    type=int,
)
parser.add_argument(
    "-n",
    "--nobstacle",
    help="number of obstacles in the range (-0.18,0.6) of theta, coefficient modifying the pose of the obstacle",
    default=20.,
    type=float,
)

args = parser.parse_args()

###* OPTIONS
WITH_DISPLAY = args.display
T = args.maxit
N_OBSTACLES = args.nobstacle
dt = 1e-3


###* LOADING THE ROBOT
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)
srdf_model_path = model_path + "/panda/demo.srdf"

###* Creating the robot
robot_wrapper = RobotWrapper(
    urdf_model_path=urdf_model_path, mesh_dir=mesh_dir, srdf_model_path=srdf_model_path
)
rmodel, cmodel, vmodel = robot_wrapper()
rdata = rmodel.createData()

# Initial config
INITIAL_CONFIG = pin.neutral(rmodel)

###* CREATING THE TARGET
TARGET_SHAPE = hppfcl.Sphere(5e-2)
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 1]))
TARGET_POSE = pin.SE3(pin.utils.rotate('x',np.pi), np.array([0., -0.2, 1]))

###* ADDING THE OBSTACLE
OBSTACLE_RADIUS = 5e-2
OBSTACLE_SHAPE = hppfcl.Sphere(OBSTACLE_RADIUS)

OBSTACLE_POSE = TARGET_POSE.copy()
OBSTACLE_POSE.translation = np.array([.2, 0, 1.5])

###* DISPLACING THE OBSTACLE THROUGH THETA
theta_list = np.arange(-0.2, 0, 0.2 / N_OBSTACLES)
#Initial X0
q0 = INITIAL_CONFIG
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

# For  the WARM START
X0_WS = [x0 for i in range(T+1)]


# SETTING UP THE VISUALIZER
MeshcatVis = MeshcatWrapper()
vis, meshcatvis = MeshcatVis.visualize(
    TARGET_POSE,
    OBSTACLE_POSE,
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
    obstacle_type="sphere",
    OBSTACLE_DIM=OBSTACLE_RADIUS,
)
vis.display(q0)

###* ADDING THE OBSTACLE IN THE COLLISION MODEL
#! parentJoint not implemented in pino2 but deprec. in pino3.
if "2.6" in pin.__version__:
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE_SHAPE,
        OBSTACLE_POSE,
    )
if "2.9" in pin.__version__:
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
        OBSTACLE_SHAPE,
        OBSTACLE_POSE,
    )
IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

###* ADDING THE COLLISION PAIRS 
for k in range(16, 26):
    cmodel.addCollisionPair(pin.CollisionPair(k, IG_OBSTACLE))

cdata = cmodel.createData()

# Storing the q through the evolution of theta
list_Q_theta = []
list_Q_theta_ws = []
FIRST_IT = True
###* GOING THROUGH ALL THE THETA AND SOLVING THE OCP FOR EACH THETA
for theta in theta_list:
    print(f"theta = {theta}")

    # Generate a reachable obstacle
    # OBSTACLE_POSE.translation = TARGET_POSE.translation / 2 + [
    #     0.2 + theta,
    #     0 + theta,
    #     1.0 + theta,
    # ]
    OBSTACLE_POSE.translation -= np.array([0,theta, 0])
    
    
    # Updating the pose of the obstacle in the collision model
    cmodel.geometryObjects[cmodel.getGeometryId("obstacle")].placement = OBSTACLE_POSE
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q0)

    # CREATING THE PROBLEM WITHOUT WARM START
    problem = OCPPandaReachingCol(
        rmodel,
        cmodel,
        TARGET_POSE,
        T,
        dt,
        x0,
        WEIGHT_GRIPPER_POSE=100,
        WEIGHT_COL=10000,
        WEIGHT_uREG=1e-4,
        WEIGHT_xREG=1e-1,
    )
    ddp = problem()
    # Solving the problem
    xx = ddp.solve()

    log = ddp.getCallbacks()[0]

    Q_sol = []
    for xs in log.xs:
            Q_sol.append(np.array(xs[:7].tolist()))
    list_Q_theta.append(Q_sol)
    

    # CREATING THE PROBLEM WITH WARM START
    problem = OCPPandaReachingCol(
        rmodel,
        cmodel,
        TARGET_POSE,
        T,
        dt,
        x0,
        WEIGHT_GRIPPER_POSE=100,
        WEIGHT_COL=10000,
        WEIGHT_uREG=1e-4,
        WEIGHT_xREG=1e-1,
    )
    ddp = problem()
    
    if FIRST_IT:
        U0_WS = ddp.problem.quasiStatic(X0_WS[:-1])
        FIRST_IT = False
    # Solving the problem
    xx = ddp.solve(X0_WS, U0_WS)

    log = ddp.getCallbacks()[0]

    Q_sol = []
    for xs in log.xs:
            Q_sol.append(np.array(xs[:7].tolist()))
    list_Q_theta_ws.append(Q_sol)
    
    X0_WS = log.xs
    U0_WS = log.us

###* DISPLAYING THE RESULTS IN MESHCAT

# Removing the obstacle of the geometry model because 
cmodel.removeGeometryObject("obstacle")

for k  in range(len(theta_list)):
    theta = theta_list[k]
    print(f"press enter for displaying the {k}-th trajectory where theta = {theta}")
    input()
    Q = list_Q_theta[k]
    Q_WS = list_Q_theta_ws[k]
    # OBSTACLE_POSE.translation = TARGET_POSE.translation / 2 + [
    #     0.2 + theta,
    #     0 + theta,
    #     1.0 + theta,
    # ]
    OBSTACLE_POSE.translation -= np.array([0,theta,0])
    
    meshcatvis["obstacle"].set_transform(get_transform(OBSTACLE_POSE))

    for q in Q:
        vis.display(q)
        input()
    
    print("Now same k but with a warm start")
    for q in Q_WS:
        vis.display(q)
        input()