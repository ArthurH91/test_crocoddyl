from os.path import dirname, join, abspath

import hppfcl 
import numpy as np
import pinocchio as pin

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from ocp_panda_reaching_col import OCPPandaReachingCol

from utils import get_transform, BLUE, YELLOW
from utils_plot import display_with_col



### HYPERPARMS

T = 100
WEIGHT_TERM_POS=1
WEIGHT_COL=10
WEIGHT_XREG=1e-4
WEIGHT_UREG=1e-6


DT = 1/T

### CREATING THE TARGET
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 1.55]))
TARGET_POSE.translation = np.array([0, -0.4,1.5])



###* LOADING THE ROBOT
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)
srdf_model_path = model_path + "/panda/demo.srdf"

###* Creating the robot
robot_wrapper = RobotWrapper(
    urdf_model_path=urdf_model_path, mesh_dir=mesh_dir, srdf_model_path=srdf_model_path, auto_col=True
)
rmodel, cmodel, vmodel = robot_wrapper()
rdata = rmodel.createData()

### INITIAL CONFIG OF THE ROBOT
INITIAL_CONFIG = pin.neutral(rmodel)

### ADDING THE OBSTACLE
OBSTACLE_RADIUS = 8e-2

OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)
OBSTACLE_POSE = pin.SE3.Identity()
OBSTACLE_POSE.translation = np.array([0.25, -0.500, 1.5])

###* DISPLACING THE OBSTACLE THROUGH THETA
# theta_list = np.arange(-0.4, 0, 0.2 / N_OBSTACLES)

start = 0
stop = 0.05
step = 0.005
theta_list = np.arange(start,stop, step)

#Initial X0
x0 = np.concatenate([INITIAL_CONFIG, pin.utils.zero(rmodel.nv)])

# For  the WARM START
X0_WS = [x0 for i in range(T+1)]

OBSTACLE_POSE.translation += np.array([0,start,0])
    

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

# Storing the q through the evolution of theta
list_Q_theta = []
list_Q_theta_ws = []
FIRST_IT = True
###* GOING THROUGH ALL THE THETA AND SOLVING THE OCP FOR EACH THETA
for k, theta in enumerate(theta_list):
    print(f"#################################################### ITERATION n°{k} out of {len(theta_list)-1}####################################################")
    print(f"theta = {round(theta,3)} , step = {round(theta_list[0]-theta_list[1], 3)}, theta min = {round(theta_list[0],3)}, theta max = {round(theta_list[-1],3)} ")
    # Generate a reachable obstacle
    OBSTACLE_POSE.translation += np.array([0,start,0])

    
    # Updating the pose of the obstacle in the collision model
    cmodel.geometryObjects[cmodel.getGeometryId("obstacle")].placement = OBSTACLE_POSE
    pin.framesForwardKinematics(rmodel, rdata, INITIAL_CONFIG)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, INITIAL_CONFIG)

    # CREATING THE PROBLEM WITHOUT WARM START
    problem = OCPPandaReachingCol(
        rmodel,
        cmodel,
        TARGET_POSE,
        T,
        DT,
        x0,
        WEIGHT_GRIPPER_POSE=WEIGHT_TERM_POS,
        WEIGHT_COL=WEIGHT_COL,
        WEIGHT_uREG=WEIGHT_UREG,
        WEIGHT_xREG=WEIGHT_XREG,
    )
    ddp = problem()
    # Solving the problem
    xx = ddp.solve()

    Q_sol = []
    for xs in ddp.xs:
            Q_sol.append(np.array(xs[:7].tolist()))
    list_Q_theta.append(Q_sol)
    
    print("######################################### WARM START ##############################################")

    # CREATING THE PROBLEM WITH WARM START
    problem = OCPPandaReachingCol(
        rmodel,
        cmodel,
        TARGET_POSE,
        T,
        DT,
        x0,
        WEIGHT_GRIPPER_POSE=WEIGHT_TERM_POS,
        WEIGHT_COL=WEIGHT_COL,
        WEIGHT_uREG=WEIGHT_UREG,
        WEIGHT_xREG=WEIGHT_XREG,
    )
    ddp = problem()
    # Solving the problem
    if FIRST_IT:
        xx = ddp.solve()
        FIRST_IT = False
    else:
        xx = ddp.solve(X0_WS, U0_WS)

    Q_sol = []
    for xs in ddp.xs:
            Q_sol.append(np.array(xs[:7].tolist()))
    list_Q_theta_ws.append(Q_sol)
    
    X0_WS = ddp.xs
    U0_WS = ddp.us

    print(f"---------------------------------------------------------------------------------------------------------------------------")
###* DISPLAYING THE RESULTS IN MESHCAT

# Removing the obstacle of the geometry model because 
OBSTACLE_POSE.translation += np.array([0,start,0])

cmodel.geometryObjects[cmodel.getGeometryId("obstacle")].placement = OBSTACLE_POSE
pin.framesForwardKinematics(rmodel, rdata, INITIAL_CONFIG)
pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, INITIAL_CONFIG)
# SETTING UP THE VISUALIZER
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
vis.display(INITIAL_CONFIG)

while True:
    for k  in range(len(theta_list)):
        theta = theta_list[k]
        print(f"press enter for displaying the {k}-th trajectory where theta = {round(theta,3)}")
        input()

        Q = list_Q_theta[k]
        Q_WS = list_Q_theta_ws[k]
        
        OBSTACLE_POSE.translation += np.array([0,theta,0])
        
        cmodel.geometryObjects[cmodel.getGeometryId("obstacle")].placement = OBSTACLE_POSE
        pin.framesForwardKinematics(rmodel, rdata, INITIAL_CONFIG)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, INITIAL_CONFIG)
        meshcatvis["obstacle"].set_transform(get_transform(OBSTACLE_POSE))

        display_with_col(Q, vis, meshcatvis, rmodel, rdata, cmodel, cdata)
        print("Press enter to get rid of the little green dots")
        input()  
        for k in range(T):
            for i in range(len(cmodel.collisionPairs)):
                meshcatvis["cp"+ str(i) + str(k)].delete()
        print("Now press enter for the same k but with a warm start")
        input()
        display_with_col(Q_WS, vis, meshcatvis, rmodel, rdata, cmodel, cdata)
        print("Press enter to get rid of the little green dots")
        input()
        for k in range(T):
            for i in range(len(cmodel.collisionPairs)):
                meshcatvis["cp"+ str(i) + str(k)].delete()

    input("replay?")