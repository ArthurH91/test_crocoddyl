import numpy as np
import pinocchio as pin
import hppfcl

scenario = "big_obstacle"

def chose_scenario(scenario = "big_obstacle"):
    """Choses the scenario optimized. 


    Args:
        scenario (str, optional): Scenario to be optimized. Defaults to "big_obstacle".

    Returns:
        T (int): Number of nodes.
        WEIGHT_XREG (float): Weight of the X regularization.
        WEIGHT_UREG (float): Weight of the U regularization.
        WEIGHT_TERM_POS (float): Weight of the distance of the end effector to the target at the terminal configuration.
        WEIGHT_COL (float): Weight of the collision in the running cost.
        WEIGHT_TERM_COL (float): Weight of the collision in the terminal cost.
        MAXIT (int): Number of max iterations of the solver.
        TARGET_POSE (pin.SE3): Target pose in pin.SE3.
        OBSTACLE_DIM (np.ndarray) : Dimension of the obstacle.
        OBSTACLE (hppfcl.ShapeBase) : Hppfcl shape of the obstacle.
        OBSTACLE_POSE (pin.SE3): Obstacle pose in pin.SE3.
        INITIAL_CONFIG (np.ndarray) : Initial configuration of the robot.
    """
    if scenario == "big_obstacle":
        
        # Number of nodes
        T = 500

        # Weights in the solver
        WEIGHT_XREG = 1e-3
        WEIGHT_UREG = 1e-3
        WEIGHT_TERM_POS = 100
        WEIGHT_COL = 1
        WEIGHT_TERM_COL = 1
        

        # Number max of iterations in the solver
        MAXIT = 100
        
        # Target pose
        TARGET = np.array([-0.05, 0., 1.056])
        TARGET_POSE = pin.SE3.Identity()
        TARGET_POSE.translation = TARGET
        TARGET_POSE.rotation = pin.utils.rotate("x", np.pi)

        # Creation of the obstacle
        # OBSTACLE_DIM = np.array([1e-2, 8e-1,8e-1])
        OBSTACLE_DIM = 3e-1
        # OBSTACLE = hppfcl.Box(OBSTACLE_DIM)
        OBSTACLE = hppfcl.Sphere(OBSTACLE_DIM)
        OBSTACLE_POSE = pin.SE3.Identity()
        OBSTACLE_POSE.translation = np.array([0, 0.327, 1.2])
        
        INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0,0])
        
    
    return T, WEIGHT_XREG, WEIGHT_UREG, WEIGHT_TERM_POS, WEIGHT_COL, WEIGHT_TERM_COL, MAXIT, TARGET_POSE, OBSTACLE_DIM, OBSTACLE, OBSTACLE_POSE, INITIAL_CONFIG