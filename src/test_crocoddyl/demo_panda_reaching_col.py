from os.path import dirname, join, abspath
import argparse
import time

import numpy as np
import pinocchio as pin
import hppfcl

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from ocp_panda_reaching_col import OCPPandaReachingCol

from utils import BLUE, YELLOW

###* PARSERS
parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--display", help="display the results", action="store_true", default=False
)


args = parser.parse_args()

###* OPTIONS
WITH_DISPLAY = args.display
T = 40

dt = 2/T

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
rdata = rmodel.createData()

### CREATING THE TARGET
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 1.55]))
TARGET_POSE.translation = np.array([0, -0.4,1.5])

### INITIAL CONFIG OF THE ROBOT
INITIAL_CONFIG = pin.neutral(rmodel)

### ADDING THE OBSTACLE
OBSTACLE_RADIUS = 8e-2

OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)
OBSTACLE_POSE = pin.SE3.Identity()
OBSTACLE_POSE.translation = np.array([0.25, -0.45, 1.5])


OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
    "obstacle",
    rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
    rmodel.getFrameId("universe"),
    OBSTACLE_POSE,
    OBSTACLE,
)
OBSTACLE_GEOM_OBJECT.meshColor = YELLOW

### ADDING THE COLLISION PAIRS TO THE COLLISION MODEL
IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)
end_effector_id = cmodel.getGeometryId("panda2_link5_capsule28")
cmodel.addCollisionPair(pin.CollisionPair(end_effector_id, IG_OBSTACLE))
cmodel.geometryObjects[end_effector_id].meshColor = BLUE

cdata = cmodel.createData()

# Generating the meshcat visualizer
if WITH_DISPLAY:
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        TARGET_POSE,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
    )
    vis = vis[0]
    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)

### INITIAL X0
q0 = INITIAL_CONFIG
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

### CREATING THE PROBLEM
problem = OCPPandaReachingCol(
    rmodel,
    cmodel,
    TARGET_POSE,
    T,
    dt,
    x0,
    WEIGHT_GRIPPER_POSE=1,
    WEIGHT_COL=1e4 * 0,
    WEIGHT_xREG=1e-1,
    WEIGHT_uREG=1e-4,
)
ddp = problem()
# Solving the problem
XS_init = [x0] * (T+1)
US_init = [np.zeros(rmodel.nv)] * T 
US_init = ddp.problem.quasiStatic(XS_init[:-1])
ddp.solve(XS_init, US_init, 400)

print("End of the computation, press enter to display the traj if requested.")
### DISPLAYING THE TRAJ
Q = []
while True:
    vis.display(INITIAL_CONFIG)
    input()
    for xs in ddp.xs:
        vis.display(np.array(xs[:7].tolist()))
        Q.append(xs[:7].tolist())
        # time.sleep(1e-)
        input()
    print("replay?")
