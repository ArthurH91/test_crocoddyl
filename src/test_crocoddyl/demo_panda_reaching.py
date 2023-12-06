from os.path import dirname, join, abspath
import argparse
import time
import json,codecs

import numpy as np
import pinocchio as pin

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
    "-step",
    "--step",
    help="number of steps in the trajectory",
    default=200,
    type=int,
)
parser.add_argument(
    "-s",
    "--save",
    help="save the results in a Json file",
    action="store_true",
    default=False,
)
args = parser.parse_args()

###* OPTIONS
WITH_DISPLAY = args.display
T = args.step
WITH_SAVING = args.save

dt = 1e-3

name_file = ""

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
cdata = cmodel.createData()

### CREATING THE TARGET
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 1.55]))
TARGET_POSE.translation = np.array([0, -0.4,1.5])

### INITIAL CONFIG OF THE ROBOT
INITIAL_CONFIG = pin.neutral(rmodel)

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


################################################################ WARM START ############################################################

###* WARM START 

results_json = codecs.open(
   "results_test" + ".json", "r", encoding="utf-8"
).read()
# Loading the json file
results = json.loads(results_json)
Q_WS = results["Q"]

### CREATING THE PROBLEM
problem = OCPPandaReachingCol(
    rmodel, cmodel, TARGET_POSE, T, dt, x0, WEIGHT_GRIPPER_POSE=100, WEIGHT_xREG=1e-1
)
ddp = problem()
# Solving the problem
xs_ws = [np.array(x) for x in Q_WS]
ddp.solve()

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
        "Q": Q,
        "target_pose": get_transform(TARGET_POSE).tolist(),
    }
    with open(
        dirname(str(abspath(__file__))) + "/results/" + name_file + ".json", "w"
    ) as outfile:
        json.dump(results, outfile)
