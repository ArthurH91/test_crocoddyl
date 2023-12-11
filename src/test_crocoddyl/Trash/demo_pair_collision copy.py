from os.path import dirname, join, abspath
import time
import numpy as np
import pinocchio as pin
import argparse

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper

from ocp_pair_collision import OCPPandaReachingCol

from scenario import chose_scenario
from utils import BLUE
from utils_plot import plot_costs, display_with_col
### PARSERS
parser = argparse.ArgumentParser(description="Parser to select the scenario.")

parser_group = parser.add_mutually_exclusive_group()
parser_group.add_argument("-bigb", action="store_const", const="big_ball", dest="scenario", help="Set up the scenario to the big ball one.")
parser_group.add_argument("-smallb", action="store_const", const="small_ball", dest="scenario", help="Set up the scenario to the small ball one.")
parser_group.add_argument("-bigw", action="store_const", const="big_wall", dest="scenario", help="Set up the scenario to the big wall one.")
parser_group.add_argument("-smallw", action="store_const", const="small_wall", dest="scenario", help="Set up the scenario to the small wall one.")
parser_group.add_argument("-smallwf", action="store_const", const="small_wall_floor", dest="scenario", help="Set up the scenario to the small wall one.")
parser_group.add_argument("-debug", action="store_const", const="debug", dest="scenario", help="Set up the debug scenario")

args = parser.parse_args()

scenario = args.scenario

if scenario is None:
    scenario = "small_ball_sliding"
print(f"Scenario : {scenario}")

### HYPERPARMS

(
    T,
    WEIGHT_XREG,
    WEIGHT_UREG,
    WEIGHT_TERM_POS,
    WEIGHT_COL,
    WEIGHT_TERM_COL,
    MAXIT,
    TARGET_POSE,
    OBSTACLE_DIM,
    OBSTACLE,
    OBSTACLE_POSE,
    INITIAL_CONFIG,
    DT,
    RUNNING_COST_ENDEFF
) = chose_scenario(scenario)

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
    urdf_model_path=urdf_model_path, mesh_dir=mesh_dir, srdf_model_path=srdf_model_path
)
rmodel, cmodel, vmodel = robot_wrapper()
rdata = rmodel.createData()


# Initial trajectory
Q0 = np.concatenate([INITIAL_CONFIG] * (T + 1))

# Initial vectors for the solver
x0 = np.concatenate(
    [INITIAL_CONFIG, pin.utils.zero(rmodel.nv)]
)  # Initial configuration + no speed
X0 = [x0] * (T + 1)  # Initial guess
u0 = np.zeros(7)  # Initial command
U0 = [u0] * T

### COLLISIONS
# Adding the obstacle to the collision model
OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
    "obstacle",
    rmodel.getFrameId("universe"),
    rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
    OBSTACLE,
    OBSTACLE_POSE,
)
OBSTACLE_GEOM_OBJECT.meshcolor = BLUE
OBSTACLE_GEOM_OBJECT.meshColor = BLUE

IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

# Adding the collision pairs to the model
for k in range(16, 20):
    cmodel.addCollisionPair(pin.CollisionPair(k, IG_OBSTACLE))
cdata = cmodel.createData()

### SOLVING THE OCP
problem = OCPPandaReachingCol(
    rmodel,
    cmodel,
    cdata,
    TARGET_POSE,
    T,
    DT,
    x0,
    WEIGHT_TERM_POS=WEIGHT_TERM_POS,
    WEIGHT_COL=WEIGHT_COL,
    WEIGHT_TERM_COL=WEIGHT_TERM_COL,
    WEIGHT_UREG=WEIGHT_UREG,
    WEIGHT_XREG=WEIGHT_XREG,
    RUNNING_COST_ENDEFF=RUNNING_COST_ENDEFF
)
ddp = problem()
ddp.solve(X0, U0, maxiter=MAXIT)

log = ddp.getCallbacks()

### VISUALIZING THE RESULT
# Creation of the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis, meshcatvis = MeshcatVis.visualize(
    TARGET_POSE,
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
)
vis = vis
# Displaying the initial configuration of the robot
vis.display(INITIAL_CONFIG)

Q = []
for xs in ddp.xs:
    Q.append(np.array(xs[:7].tolist()))
rd = ddp.problem.runningDatas
plot_costs(rd)

input()
while True:
    display_with_col(Q, vis, meshcatvis, rmodel, rdata, cmodel, cdata)
    input("replay?")
input()
# while True:
#     for xs in ddp.xs:
#         vis.display(np.array(xs[:7].tolist()))
#         time.sleep(1e-3)
#         Q.append(np.array(xs[:7].tolist()))
#         # input()
#     input("Replay?")

# Q_flat_list = [item for sublist in Q for item in sublist]
# results = {
#     "Q": Q_flat_list,
# }
# with open("results_to_evaluate.json", "w") as outfile:
#     json.dump(results, outfile)
