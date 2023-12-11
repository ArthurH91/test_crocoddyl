import numpy as np
import pinocchio as pin
import hppfcl


def chose_scenario(scenario="big_obstacle"):
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
        DT (float) : Time step.
    """
    DT = 1e-3
    if scenario == "big_ball":
        # Number of nodes
        T = 500

        # Weights in the solver
        WEIGHT_XREG = 1e-1
        WEIGHT_UREG = 1e-3
        WEIGHT_TERM_POS = 100
        WEIGHT_COL = 5
        WEIGHT_TERM_COL = 5
        WEIGHT_LIMIT = 5

        # Number max of iterations in the solver
        MAXIT = 200

        # Target pose
        TARGET = np.array([-0.05, 0.0, 1.056])
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

        INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0, 0])
        RUNNING_COST_ENDEFF = True

        print(
            """---------------------------------------------------------------\n Small collisions at the end but collision avoidance otherwise. 
              \n --------------------------------------------------------------"""
        )

    
    elif scenario == "small_ball":
        # Number of nodes
        T = 500

        # Weights in the solver
        WEIGHT_XREG = 1e-1
        WEIGHT_UREG = 1e-2
        WEIGHT_TERM_POS = 100
        WEIGHT_COL = 1000
        WEIGHT_TERM_COL = 1000
        WEIGHT_LIMIT = 10
        # Number max of iterations in the solver
        MAXIT = 100

        # Target pose
        TARGET = np.array([-0.05, - 0.2, 1.056])
        TARGET_POSE = pin.SE3.Identity()
        TARGET_POSE.translation = TARGET
        TARGET_POSE.rotation = pin.utils.rotate("x", np.pi)

        # Creation of the obstacle
        OBSTACLE_DIM = 1e-1
        OBSTACLE = hppfcl.Sphere(OBSTACLE_DIM)
        OBSTACLE_POSE = pin.SE3.Identity()
        OBSTACLE_POSE.translation = np.array([0.15, -.07, 1.2])

        INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0, 0])

        RUNNING_COST_ENDEFF = True
        print(
            """---------------------------------------------------------------\n Small collisions at the end but collision avoidance otherwise. 
              \n --------------------------------------------------------------"""
        )

    elif scenario == "debug":
        # Number of nodes
        T = 500

        # Weights in the solver
        WEIGHT_XREG = 1e-2
        WEIGHT_UREG = 1e-3
        WEIGHT_TERM_POS = 100
        WEIGHT_COL = 1e20
        WEIGHT_TERM_COL = 1e20
        WEIGHT_LIMIT = 1

        # Number max of iterations in the solver
        MAXIT = 1

        # Target pose
        TARGET = np.array([0, -0.2, 1.0])
        TARGET_POSE = pin.SE3.Identity()
        TARGET_POSE.translation = TARGET
        TARGET_POSE.rotation = pin.utils.rotate("x", np.pi)

        # Creation of the obstacle
        OBSTACLE_DIM = 3e-1
        OBSTACLE = hppfcl.Sphere(OBSTACLE_DIM)
        OBSTACLE_POSE = pin.SE3.Identity()
        OBSTACLE_POSE.translation = np.array([0.15, 0.3, 1.2])

        # theta = -0.1
        # OBSTACLE_POSE.translation = TARGET_POSE.translation / 2 + [
        #     0.2 + theta,
        #     0 + theta,
        #     1.0 + theta,
        # ]

        INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0, 0])
        RUNNING_COST_ENDEFF = True

        print(
            """---------------------------------------------------------------\n DEBUG 
              \n --------------------------------------------------------------"""
        )

    elif scenario == "small_ball_sliding":
        # Number of nodes
        T = 500

        # Weights in the solver
        WEIGHT_XREG = 1e-1
        WEIGHT_UREG = 1e-2
        WEIGHT_TERM_POS = 100
        WEIGHT_COL = 1
        WEIGHT_TERM_COL = 1
        WEIGHT_LIMIT = 1

        # Number max of iterations in the solver
        MAXIT = 100

        # Target pose
        TARGET = np.array([0, -0.2, 1.0])
        TARGET_POSE = pin.SE3.Identity()
        TARGET_POSE.translation = TARGET
        # TARGET_POSE.rotation = pin.utils.rotate("x", np.pi)

        # Creation of the obstacle
        OBSTACLE_DIM = 1e-1
        OBSTACLE = hppfcl.Sphere(OBSTACLE_DIM)
        OBSTACLE_POSE = pin.SE3.Identity()
        OBSTACLE_POSE.translation = np.array([0.1, -0.5, 1.2])

        INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0, 0])
        RUNNING_COST_ENDEFF = True

    elif scenario == "big_wall":
        # Number of nodes
        T = 500

        # Weights in the solver
        WEIGHT_XREG = 1e-1
        WEIGHT_UREG = 1e-3
        WEIGHT_TERM_POS = 25
        WEIGHT_COL = 2
        WEIGHT_TERM_COL = 2
        WEIGHT_LIMIT = 1

        # Number max of iterations in the solver
        MAXIT = 100

        # Target pose
        TARGET = np.array([-0.05, -0.15, 1.056])
        TARGET_POSE = pin.SE3.Identity()
        TARGET_POSE.translation = TARGET
        TARGET_POSE.rotation = pin.utils.rotate("x", np.pi)

        # Creation of the obstacle
        OBSTACLE_DIM = np.array([1e-2, 8e-1, 8e-1])
        OBSTACLE = hppfcl.Box(OBSTACLE_DIM)
        OBSTACLE_POSE = pin.SE3.Identity()
        OBSTACLE_POSE.translation = np.array([0.15, 0.307, 1.2])

        INITIAL_CONFIG = np.array([0.5, 0.5, 0, 0, 0, 0, 0])
        INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0, 0])      
        # INITIAL_CONFIG = np.array([0.5,0.5,0,0,0,0.5,0])

        RUNNING_COST_ENDEFF = True
        print(
            """---------------------------------------------------------------\n Completely avoids collision but doesn't go completely to the target. 
              \n --------------------------------------------------------------"""
        )
    elif scenario == "small_wall":
        # Number of nodes
        T = 500

        # Weights in the solver
        WEIGHT_XREG = 5e-1
        WEIGHT_UREG = 1e-3
        WEIGHT_TERM_POS = 100
        WEIGHT_COL = 10000
        WEIGHT_TERM_COL = 10000
        WEIGHT_LIMIT = 1

        # Number max of iterations in the solver
        MAXIT = 100

        # Target pose
        TARGET = np.array([-0.1, 0, 0.85])
        TARGET_POSE = pin.SE3.Identity()
        TARGET_POSE.translation = TARGET
        TARGET_POSE.rotation = pin.utils.rotate("x", np.pi)

        # Creation of the obstacle
        OBSTACLE_DIM = np.array([1e-2, 2e-1, 3e-1])
        OBSTACLE = hppfcl.Box(OBSTACLE_DIM)
        OBSTACLE_POSE = pin.SE3.Identity()
        OBSTACLE_POSE.translation = np.array([0.07, 0.0, 0.85])

        # INITIAL_CONFIG = np.array([0, 0, 0, -0.5, 0, 1, 0])
        INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0, 0])

        RUNNING_COST_ENDEFF = True
        print(
            """---------------------------------------------------------------\n Avoids collision but touches slightly the wall. 
              \n --------------------------------------------------------------"""
        )
    elif scenario == "small_wall_floor":
        # Number of nodes
        T = 500

        # Weights in the solver
        WEIGHT_XREG = 1e-3
        WEIGHT_UREG = 1e-2
        WEIGHT_TERM_POS = 1e2
        WEIGHT_COL = 1e6
        WEIGHT_TERM_COL = 1e6

        # Number max of iterations in the solver
        MAXIT = 200

        # Target pose
        TARGET = np.array([0, 0, 0.85])
        TARGET_POSE = pin.SE3.Identity()
        TARGET_POSE.translation = TARGET
        TARGET_POSE.rotation = pin.utils.rotate("x", np.pi)

        # Creation of the obstacle
        OBSTACLE_DIM = np.array([1e-2, 3e-1, 3e-1])
        OBSTACLE = hppfcl.Box(OBSTACLE_DIM)
        OBSTACLE_POSE = pin.SE3.Identity()
        OBSTACLE_POSE.translation = np.array([0.32, 0.0, 0.9])

        # INITIAL_CONFIG = np.array([0, 0, 0, -0.5, 0, 1, 0])
        INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0, 0])
        # INITIAL_CONFIG = np.array([0.0, 0.5, 0.0, -0.2, 0.0, 0.0, 0.0])
        RUNNING_COST_ENDEFF = True
        print(
            """---------------------------------------------------------------\n Avoids collision but touches slightly the wall. 
              \n --------------------------------------------------------------"""
        )
    else:
        raise ValueError("The name of this scenario is not defined.")
    return (
        T,
        WEIGHT_XREG,
        WEIGHT_UREG,
        WEIGHT_TERM_POS,
        WEIGHT_COL,
        WEIGHT_TERM_COL,
        WEIGHT_LIMIT,
        MAXIT,
        TARGET_POSE,
        OBSTACLE_DIM,
        OBSTACLE,
        OBSTACLE_POSE,
        INITIAL_CONFIG,
        DT,
        RUNNING_COST_ENDEFF,
    )
