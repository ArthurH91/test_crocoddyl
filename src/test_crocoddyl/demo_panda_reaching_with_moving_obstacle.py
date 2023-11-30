from os.path import dirname, join, abspath
import argparse
import time 

import numpy as np
import pinocchio as pin

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from ocp_pair_collision import OCPPandaReachingCol
from scenario import chose_scenario

from utils import get_transform



###* PARSERS
parser = argparse.ArgumentParser()

parser.add_argument(
    "-n",
    "--nobstacle",
    help="number of obstacles in the range (-0.18,0.6) of theta, coefficient modifying the pose of the obstacle",
    default=20.,
    type=float,
)

args = parser.parse_args()

###* OPTIONS
N_OBSTACLES = args.nobstacle

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
) = chose_scenario("small_ball_sliding")


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

###* DISPLACING THE OBSTACLE THROUGH THETA
theta_list = np.arange(-0.4, 0, 0.2 / N_OBSTACLES)
theta_list = np.arange(-0.28, -0.20, 0.005)

#Initial X0
x0 = np.concatenate([INITIAL_CONFIG, pin.utils.zero(rmodel.nv)])

# For  the WARM START
X0_WS = [x0 for i in range(T+1)]



OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
    "obstacle",
    rmodel.getFrameId("universe"),
    rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
    OBSTACLE,
    OBSTACLE_POSE,
)
IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

###* ADDING THE COLLISION PAIRS 
for k in range(16, 23):
    cmodel.addCollisionPair(pin.CollisionPair(k, IG_OBSTACLE))

cdata = cmodel.createData()

# Storing the q through the evolution of theta
list_Q_theta = []
list_Q_theta_ws = []
FIRST_IT = True
###* GOING THROUGH ALL THE THETA AND SOLVING THE OCP FOR EACH THETA
for k, theta in enumerate(theta_list):
    print(f"#################################################### ITERATION n°{k} out of {len(theta_list)-1}####################################################")
    print(f"theta = {round(theta,3)} , step = {round(theta_list[0]-theta_list[1], 3)}, theta min = {round(theta_list[0],3)}, theta max = {round(theta_list[-1],3)} ")
    # Generate a reachable obstacle
    OBSTACLE_POSE.translation = TARGET_POSE.translation / 2 + [
        0.2 + theta,
        0 + theta,
        1.0 + theta,
    ]    
    
    # Updating the pose of the obstacle in the collision model
    cmodel.geometryObjects[cmodel.getGeometryId("obstacle")].placement = OBSTACLE_POSE
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, INITIAL_CONFIG)

    # CREATING THE PROBLEM WITHOUT WARM START
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
        RUNNING_COST_ENDEFF= RUNNING_COST_ENDEFF
    )
    ddp = problem()
    # Solving the problem
    xx = ddp.solve()

    log = ddp.getCallbacks()[0]

    Q_sol = []
    for xs in log.xs:
            Q_sol.append(np.array(xs[:7].tolist()))
    list_Q_theta.append(Q_sol)
    
    print("######################################### WARM START ##############################################")

    # CREATING THE PROBLEM WITH WARM START
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

    print(f"---------------------------------------------------------------------------------------------------------------------------")
###* DISPLAYING THE RESULTS IN MESHCAT

# Removing the obstacle of the geometry model because 
cmodel.removeGeometryObject("obstacle")

# SETTING UP THE VISUALIZER
MeshcatVis = MeshcatWrapper()
vis, meshcatvis = MeshcatVis.visualize(
    TARGET_POSE,
    OBSTACLE_POSE,
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
    obstacle_type="sphere",
    OBSTACLE_DIM=OBSTACLE_DIM,
)
vis.display(INITIAL_CONFIG)

while True:
    for k  in range(len(theta_list)):
        theta = theta_list[k]
        print(f"press enter for displaying the {k}-th trajectory where theta = {theta}")
        input()
        Q = list_Q_theta[k]
        Q_WS = list_Q_theta_ws[k]
        OBSTACLE_POSE.translation = TARGET_POSE.translation / 2 + [
            0.2 + theta,
            0 + theta,
            1.0 + theta,
        ]    
        
        meshcatvis["obstacle"].set_transform(get_transform(OBSTACLE_POSE))

        for q in Q:
            vis.display(q)
            time.sleep(1e-3)  
        print("Now press enter for the same k but with a warm start")
        input()  
        for q in Q_WS:
            vis.display(q)
            time.sleep(1e-3)  
    input("replay?")