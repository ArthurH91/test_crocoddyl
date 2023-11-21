from os.path import dirname, join, abspath
import time 
import numpy as np
import pinocchio as pin
import crocoddyl
import hppfcl

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper

from ocp_pair_collision import OCPPandaReachingCol

### HYPERPARMS

# Number of nodes
T = 500

# Weights in the solver
WEIGHT_DQ = 1e-3
WEIGHT_TERM_POS = 10
WEIGHT_COL = 1
WEIGHT_TERM_COL = 1

# Number max of iterations in the solver
maxit = 100

# Target pose
TARGET = np.array([-0.25, 0., 1.056])

TARGET_POSE = pin.SE3.Identity()
TARGET_POSE.translation = TARGET


### CREATION OF THE ROBOT
# Getting the urdf & srdf files 
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)
srdf_model_path = model_path + "/panda/demo.srdf"

# Creating the robot
robot_wrapper = RobotWrapper(
    urdf_model_path=urdf_model_path,
    mesh_dir=mesh_dir,
    srdf_model_path=srdf_model_path
)
rmodel, cmodel, vmodel = robot_wrapper()
rdata = rmodel.createData()

# INITIAL CONFIG OF THE ROBOT
INITIAL_CONFIG = np.array([0, -2.5, 2, -1.2, -1.7, 0,0])
INITIAL_CONFIG = pin.neutral(rmodel)

# Initial trajectory
Q0 = np.concatenate([INITIAL_CONFIG] * (T + 1))

# Initial vectors for the solver
dt = 1e-3 # Time step
x0 = np.concatenate([INITIAL_CONFIG, pin.utils.zero(rmodel.nv)]) # Initial configuration + no speed 
X0 = ([x0]*(T+1)) # Initial guess
u0 = np.zeros(7) # Initial command
U0 = ([u0] * T)

### COLLISIONS 
# Creation of the obstacle
# OBSTACLE_DIM = np.array([1e-2, 8e-1,8e-1])
OBSTACLE_DIM = 2e-1
# OBSTACLE = hppfcl.Box(OBSTACLE_DIM)
OBSTACLE = hppfcl.Sphere(OBSTACLE_DIM)
OBSTACLE_POSE = pin.SE3.Identity()
OBSTACLE_POSE.translation = np.array([.2, -0.357, 1.2])

# Adding the obstacle to the collision model
OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
    "obstacle",
    rmodel.getFrameId("universe"),
    rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
    OBSTACLE,
    OBSTACLE_POSE
)
IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

# Adding the collision pairs to the model
for k in range(16,23):
    cmodel.addCollisionPair(pin.CollisionPair(k, IG_OBSTACLE))
cdata = cmodel.createData()

### SOLVING THE OCP
problem = OCPPandaReachingCol(rmodel, cmodel, cdata, TARGET_POSE, T, dt, x0, WEIGHT_GRIPPER_POSE=WEIGHT_TERM_POS, WEIGHT_COL=10, WEIGHT_COL_TERM= WEIGHT_TERM_COL, WEIGHT_uREG=1e-4, WEIGHT_xREG=1e-1)
ddp = problem()
ddp.solve(X0, U0, maxiter = maxit)

log = ddp.getCallbacks()

### VISUALIZING THE RESULT
# Creation of the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis = MeshcatVis.visualize(
    TARGET_POSE, robot_model=rmodel, robot_collision_model=cmodel, robot_visual_model=vmodel
)
vis = vis[0]
# Displaying the initial configuration of the robot
vis.display(INITIAL_CONFIG)
input()
for xs in ddp.xs:
    vis.display(np.array(xs[:7].tolist()))
    time.sleep(1e-3)
    # input()

