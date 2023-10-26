from os.path import dirname, join, abspath

import crocoddyl
import pinocchio as pin
import numpy as np
from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from ocp_panda_reaching_col import OCPPandaReachingCol

WITHDISPLAY = False
WITHPLOT = False



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



# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.

TARGET = pin.SE3(pin.utils.rotate('x',np.pi), np.array([-0.1, 0, 0.9]))
INITIAL_CONFIG = pin.neutral(rmodel)

# Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis = MeshcatVis.visualize(
    TARGET, robot_model=rmodel, robot_collision_model=cmodel, robot_visual_model=vmodel
)
vis = vis[0]

# Displaying the initial configuration of the robot
vis.display(INITIAL_CONFIG)

q0 = INITIAL_CONFIG
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

T = 300
dt = 1e-3
problem = OCPPandaReachingCol(rmodel, cmodel, TARGET, T, dt, x0, WEIGHT_GRIPPER_POSE=100)
ddp = problem()
xx = ddp.solve()

log = ddp.getCallbacks()[0]

vis.display(INITIAL_CONFIG)
input()
for xs in log.xs:
    vis.display(np.array(xs[:7].tolist()))
    input()
