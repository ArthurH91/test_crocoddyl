from os.path import dirname, join, abspath
import time
import numpy as np
import pinocchio as pin

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper

from ocp_pair_collision import OCPPandaReachingCol

from scenario import chose_scenario

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
) = chose_scenario("small_walle")

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
dt = 1e-3  # Time step
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
IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

# Adding the collision pairs to the model
for k in range(16, 23):
    cmodel.addCollisionPair(pin.CollisionPair(k, IG_OBSTACLE))
cdata = cmodel.createData()

### SOLVING THE OCP
problem = OCPPandaReachingCol(
    rmodel,
    cmodel,
    cdata,
    TARGET_POSE,
    T,
    dt,
    x0,
    WEIGHT_TERM_POS=WEIGHT_TERM_POS,
    WEIGHT_COL=WEIGHT_COL,
    WEIGHT_TERM_COL=WEIGHT_TERM_COL,
    WEIGHT_UREG=WEIGHT_UREG,
    WEIGHT_XREG=WEIGHT_XREG,
)
ddp = problem()
ddp.solve(X0, U0, maxiter=MAXIT)

log = ddp.getCallbacks()

### VISUALIZING THE RESULT
# Creation of the meshcat visualizer
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
input()

Q = []

while True:
    for xs in ddp.xs:
        vis.display(np.array(xs[:7].tolist()))
        time.sleep(1e-3)
        Q.append(np.array(xs[:7].tolist()))
        # input()
    input("Replay?")

Q_flat_list = [item for sublist in Q for item in sublist]
results = {
    "Q": Q_flat_list,
}
# with open("results_to_evaluate.json", "w") as outfile:
#     json.dump(results, outfile)
