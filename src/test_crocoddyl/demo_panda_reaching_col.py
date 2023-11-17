from os.path import dirname, join, abspath
import argparse
import time 
import json, codecs

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
parser.add_argument("-maxit", "--maxit", help = "number max of iterations of the solver", default=200, type=int)
parser.add_argument(
    "-s", "--save", help="save the results in a Json file", action="store_true", default=False
)
args = parser.parse_args()

###* OPTIONS
WITH_DISPLAY = args.display
T = args.maxit
WITH_SAVING = args.save

dt = 1e-3

name_file = "single_wall_without_autocollision_but_obstacle_constraints_non_neutral_start_only_translation_reaching"

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

###* WARM START 

results_json = codecs.open(
   "results_t_200" + ".json", "r", encoding="utf-8"
).read()
# Loading the json file
results = json.loads(results_json)

q_dot = results["q_dot"]
Q_sol_list = results["Q_trs"]

X0_WS = []
for k in range(T+1):
    X0_WS.append(np.concatenate((np.array(Q_sol_list[rmodel.nq * k: rmodel.nq * (k+1)]), np.array(q_dot[rmodel.nq * k: rmodel.nq * (k+1)]))))

X0_WS[-1] = np.concatenate((X0_WS[-1], np.zeros(7)))

### ADDING THE OBSTACLE 
# OBSTACLE_RADIUS = 5e-2
# OBSTACLE_DIM = np.array([1e-2, 3e-1,3e-1])
OBSTACLE_DIM = np.array([1e-2, 8e-1,8e-1])

# OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)
OBSTACLE = hppfcl.Box(OBSTACLE_DIM)
OBSTACLE_POSE = pin.SE3.Identity()
OBSTACLE_POSE.translation = np.array([.2, -0.357, 1.2])
# OBSTACLE_POSE.translation = np.array([0.22, 0, 0.9])

### CREATING THE TARGET 
TARGET_POSE = pin.SE3(pin.utils.rotate('x',np.pi),np.array([0, 0, 0.85]))

### INITIAL CONFIG OF THE ROBOT
INITIAL_CONFIG = np.array([0,0.5,0,-0.1,0,1,0])
# INITIAL_CONFIG = pin.neutral(rmodel)

# parentJoint not implemented in pino2 but deprec. in pino3.
if "2.6" in pin.__version__ :  
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE,
        OBSTACLE_POSE
    )
if "2.9" in pin.__version__:
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
        OBSTACLE,
        OBSTACLE_POSE
    )

### ADDING THE COLLISION PAIRS TO THE COLLISION MODEL
IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

### ADDING THE COLLISIONS OF ONLY THE ROBOT SHAPES AGAINST THE OBSTACLE
for k in range(16,26):
    cmodel.addCollisionPair(pin.CollisionPair(k, IG_OBSTACLE))

cdata = cmodel.createData()

# Generating the meshcat visualizer
if WITH_DISPLAY:
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        TARGET_POSE, robot_model=rmodel, robot_collision_model=cmodel, robot_visual_model=vmodel
    )
    vis = vis[0]
    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)

### INITIAL X0
q0 = INITIAL_CONFIG
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

### CREATING THE PROBLEM
problem = OCPPandaReachingCol(rmodel, cmodel, TARGET_POSE, T, dt, x0, WEIGHT_GRIPPER_POSE=100, WEIGHT_COL=10, WEIGHT_uREG=1e-4, WEIGHT_xREG=1e-1)
ddp = problem()
# Solving the problem
# U0_WS = ddp.problem.quasiStatic(X0_WS[-1])

# xx = ddp.solve(X0_WS)
xx = ddp.solve()

log = ddp.getCallbacks()[0]

print("End of the computation, press enter to display the traj if requested.")
### DISPLAYING THE TRAJ
Q = []
if WITH_DISPLAY:
    vis.display(INITIAL_CONFIG)
    input()
    for xs in log.xs:
        vis.display(np.array(xs[:7].tolist()))
        Q.append(xs[:7].tolist())
        time.sleep(1e-3)

if WITH_SAVING:
    results = {
        "Q" : Q,
        "target_pose" : get_transform(TARGET_POSE).tolist(),
        "obstacle_pose" : get_transform(OBSTACLE_POSE).tolist(),
        'obstacle_dim' : OBSTACLE_DIM.tolist(),
        "type" : "single_wall",
    }
    with open(dirname(str(abspath(__file__))) + "/results/" + name_file + ".json", "w") as outfile:
            json.dump(results, outfile)

