import os
import sys
 
import crocoddyl
import pinocchio as pin
import numpy as np
import example_robot_data
from wrapper_meshcat import MeshcatWrapper
 
WITHDISPLAY = False
WITHPLOT = False
 
# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.
 
# First, let's load the Pinocchio model for the Talos arm.
robot = example_robot_data.load('ur10')
 
rmodel = robot.model
vmodel = robot.visual_model
cmodel = robot.collision_model

TARGET = pin.SE3(np.eye(3), np.array([.7, .5, .4]))
INITIAL_CONFIG = pin.neutral(rmodel)

# Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis = MeshcatVis.visualize(
    TARGET,
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel
)
vis = vis[0]

# Displaying the initial configuration of the robot
vis.display(INITIAL_CONFIG)


# # Create a cost model per the running and terminal action model.

state = crocoddyl.StateMultibody(rmodel)
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
 
# # Note that we need to include a cost model (i.e. set of cost functions) in
# # order to fully define the action model for our optimal control problem.
# # For this particular example, we formulate three running-cost functions:
# # goal-tracking cost, state and control regularization; and one terminal-cost:
# # goal cost. First, let's create the common cost functions.
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, rmodel.getFrameId("tool0"), TARGET
                                                               )


uResidual = crocoddyl.ResidualModelControl(state)
xResidual = crocoddyl.ResidualModelControl(state)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
 
# # Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
runningCostModel.addCost("xReg", xRegCost, 1e-4)
runningCostModel.addCost("uReg", uRegCost, 1e-4)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)
 
# # Next, we need to create an action model for running and terminal knots. The
# # forward dynamics (computed using ABA) are implemented
# # inside DifferentialActionModelFullyActuated.
actuationModel = crocoddyl.ActuationModelFull(state)
dt = 1e-3
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModel), dt)
runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModel), 0.)
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
 
# # For this optimal control problem, we define 1000 knots (or running action
# # models) plus a terminal knot
T = 1000
# q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1])
q0 = INITIAL_CONFIG
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
 
# # Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverDDP(problem)

solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose(),
])

# # Solving it with the DDP algorithm
x = solver.solve()
 
# # Plotting the solution and the DDP convergence
log = solver.getCallbacks()[0]


vis.display(INITIAL_CONFIG)
input()
for xs in log.xs:
    vis.display(np.array(xs[:6].tolist()))
    input()