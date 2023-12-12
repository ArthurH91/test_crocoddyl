from os.path import dirname, join, abspath
import argparse
import time

import numpy as np
import pinocchio as pin
import hppfcl

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from ocp_panda_reaching_single_col_with_bounds import OCPPandaReachingCol

from utils import BLUE, YELLOW, get_transform

###* PARSERS
parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--display", help="display the results", action="store_true", default=False
)


args = parser.parse_args()

###* OPTIONS
WEIGHT_GRIPPER_POSE = 3e3
WEIGHT_COL = 1e7
WEIGHT_xREG = 5e-2
WEIGHT_uREG = 1e-4
WEIGHT_LIMIT = 1

WITH_DISPLAY = True
T = 100

dt = 1 / T

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
TARGET_POSE.translation = np.array([0, 0.0, 1.5])

### INITIAL CONFIG OF THE ROBOT
INITIAL_CONFIG = pin.neutral(rmodel)

### ADDING THE OBSTACLE
OBSTACLE_RADIUS = 8e-2

OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)
OBSTACLE_POSE = pin.SE3.Identity()
OBSTACLE_POSE.translation = np.array([0.25, -0.2, 1.5])


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
    vis, meshcatvis = MeshcatVis.visualize(
        TARGET_POSE,
        OBSTACLE_POSE,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
        obstacle_type="sphere",
        OBSTACLE_DIM=OBSTACLE_RADIUS,
    )
    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)

### INITIAL X0
q0 = INITIAL_CONFIG
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

start = 0
stop = 1
step = 0.5
theta_list = np.arange(start, stop, step)

results = {}

for k, theta in enumerate(theta_list):
    
    print(f"#################################################### ITERATION n°{k} out of {len(theta_list)-1}####################################################")
    print(f"theta = {round(theta,3)} , step = {round(theta_list[0]-theta_list[1], 3)}, theta min = {round(theta_list[0],3)}, theta max = {round(theta_list[-1],3)} ")
    
    
    cmodel.geometryObjects[cmodel.getGeometryId("obstacle")].placement.translation += np.array(
        [0, theta, 0]
    )
    ### CREATING THE PROBLEM
    problem = OCPPandaReachingCol(
        rmodel,
        cmodel,
        TARGET_POSE,
        T,
        dt,
        x0,
        WEIGHT_GRIPPER_POSE=WEIGHT_GRIPPER_POSE,
        WEIGHT_COL=WEIGHT_COL,
        WEIGHT_xREG=WEIGHT_xREG,
        WEIGHT_uREG=WEIGHT_uREG,
        WEIGHT_LIMIT=WEIGHT_LIMIT,
    )
    ddp = problem()
    # Solving the problem
    ddp.solve()

    results[str(theta)] = ddp.xs.tolist()

# ### DISPLAYING THE TRAJ
while True:
    for k in range(len(theta_list)):
        OBSTACLE_POSE.translation += np.array(
        [0, theta_list[k], 0]
        )
        print(OBSTACLE_POSE)
        meshcatvis["obstacle"].set_transform(get_transform(OBSTACLE_POSE))

        for xs in results[str(theta_list[k])]:
            vis.display(np.array(xs[:7]))
            input()
    print("replay?")
