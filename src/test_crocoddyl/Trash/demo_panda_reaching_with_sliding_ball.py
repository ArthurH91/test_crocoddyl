from os.path import dirname, join, abspath
import argparse

import numpy as np
import pinocchio as pin
import hppfcl

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from ocp_panda_reaching_col import OCPPandaReachingCol


###* PARSERS
parser = argparse.ArgumentParser()
parser.add_argument(
    "-caps", "--capsule", help="transform the hppfcl spheres & cylinders into capsules for collision detection", action="store_true", default=True
)
parser.add_argument(
    "-d", "--display", help="display the results", action="store_true", default=False
)
parser.add_argument("-maxit", "--maxit", help = "number max of iterations of the solver", default=250, type=int)
args = parser.parse_args()

###* OPTIONS
WITH_DISPLAY = args.display
T = args.maxit
dt = 1e-3

### LOADING THE ROBOT
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
cdata = cmodel.createData()

### ADDING THE OBSTACLE 
OBSTACLE_RADIUS = 5e-2
OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)

OBSTACLE_POSE = pin.SE3.Identity()
OBSTACLE_POSE.translation = np.array([.0, 0.1, 1.])

OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
    "obstacle",
    rmodel.getFrameId("universe"),
    rmodel.frames[rmodel.getFrameId("universe")].parent,
    OBSTACLE,
    OBSTACLE_POSE
)

IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

for ig, geometry_object in enumerate(cmodel.geometryObjects):
    if isinstance(geometry_object.geometry, hppfcl.Capsule):
        cp = pin.CollisionPair(ig, IG_OBSTACLE)
        cmodel.addCollisionPair(cp)



### CREATING THE TARGET 
TARGET = pin.SE3(pin.utils.rotate('x',np.pi), np.array([-0.1, 0, 0.9]))
INITIAL_CONFIG = pin.neutral(rmodel)

# Generating the meshcat visualizer
if WITH_DISPLAY:
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        TARGET, robot_model=rmodel, robot_collision_model=cmodel, robot_visual_model=vmodel
    )
    vis = vis[0]

    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)

### INITIAL X0
q0 = INITIAL_CONFIG
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

### CREATING THE PROBLEM
problem = OCPPandaReachingCol(rmodel, cmodel, TARGET, T, dt, x0, WEIGHT_GRIPPER_POSE=100)
ddp = problem()
# Solving the problem
xx = ddp.solve()

log = ddp.getCallbacks()[0]


### DISPLAYING THE TRAJ
if WITH_DISPLAY:
    vis.display(INITIAL_CONFIG)
    input()
    for xs in log.xs:
        vis.display(np.array(xs[:7].tolist()))
        input()
