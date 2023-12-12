## Class heavily inspired by the work of Sebastien Kleff : https://github.com/machines-in-motion/minimal_examples_crocoddyl

from typing import Any
import sys
import numpy as np
import crocoddyl
import pinocchio as pin
import mim_solvers
from residualDistanceCollision import ResidualCollision
from collisionCost import CostModelPairCollision

class OCPPandaReachingCol():
    """This class is creating a optimal control problem of a panda robot reaching for a target while taking auto collisions into consideration
    """
    
    def __init__(self, rmodel : pin.Model,cmodel : pin.GeometryModel,  TARGET_POSE : pin.SE3, T : int, dt : float,  x0 : np.ndarray, WEIGHT_xREG = 1e-1, WEIGHT_uREG = 1e-4, WEIGHT_COL = 1e0, WEIGHT_GRIPPER_POSE =10, WEIGHT_LIMIT = 1) -> None:
        """Creating the class for optimal control problem of a panda robot reaching for a target while taking auto collision into consideration

        Args:
            rmodel (pin.Model): pinocchio Model of the robot
            cmodel (pin.GeometryModel): Collision model of the robot
            TARGET_POSE (pin.SE3): Pose of the target in WOLRD ref
            T (int): Number of nodes in the trajectory
            dt (float): _description_
        """
        
        self._rmodel = rmodel
        self._cmodel = cmodel
        self._TARGET_POSE = TARGET_POSE
        self._T = T
        self._dt = dt
        self._x0 = x0
        
        # Weights
        self._WEIGHT_xREG = WEIGHT_xREG
        self._WEIGHT_uREG = WEIGHT_uREG
        self._WEIGHT_COL = WEIGHT_COL
        self._WEIGHT_GRIPPER_POSE = WEIGHT_GRIPPER_POSE
        self._WEIGHT_LIMIT = WEIGHT_LIMIT
        
        # Data models
        self._rdata = rmodel.createData()
        self._cdata = cmodel.createData()
        
        
        
    def __call__(self) -> Any:
        "Setting up croccodyl OCP"
        
        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)
        
        # Running & terminal cost models
        self._runningCostModel = crocoddyl.CostModelSum(self._state)
        self._terminalCostModel = crocoddyl.CostModelSum(self._state)
        
        ### Creation of cost terms
        
        # Control Regularization cost 
        uResidual = crocoddyl.ResidualModelControl(self._state)
        uRegCost = crocoddyl.CostModelResidual(self._state, uResidual)

        # State Regularization cost 
        xResidual = crocoddyl.ResidualModelState(self._state, self._x0)
        xRegCost = crocoddyl.CostModelResidual(self._state, xResidual)

    
        # End effector frame cost
        framePlacementResidual = crocoddyl.ResidualModelFrameTranslation(
    self._state, self._rmodel.getFrameId("panda2_leftfinger"), self._TARGET_POSE.translation
)
        goalTrackingCost = crocoddyl.CostModelResidual(self._state, framePlacementResidual)


        # Collision costs
        # collision_radius = 0.2
        # activationCollision = crocoddyl.ActivationModel2NormBarrier(3, collision_radius)
        
        # for k in range(len(self._cmodel.collisionPairs)):
        #     print(self._cmodel.geometryObjects[self._cmodel.collisionPairs[k].first].name,self._cmodel.geometryObjects[self._cmodel.collisionPairs[k].second].name)
        #     residualCollision = crocoddyl.ResidualModelPairCollision(self._state, self._rmodel.nq, self._cmodel, k, 7)
        #     costCollision = crocoddyl.CostModelResidual(self._state, activationCollision, residualCollision)
        #     self._runningCostModel.addCost("collision" + str(k), costCollision, self._WEIGHT_COL)
        #     self._terminalCostModel.addCost("collision_term"+ str(k), costCollision, self._WEIGHT_COL)
        self._runningConstraintModelManager = crocoddyl.ConstraintModelManager(self._state, self._actuation.nu)
        self._terminalConstraintModelManager = crocoddyl.ConstraintModelManager(self._state, self._actuation.nu)




    # Collision residuals
        for k in range(len(self._cmodel.collisionPairs)):
            # print(f"name : self._cmodel.collisionPairs[k].first : {self._cmodel.geometryObjects[self._cmodel.collisionPairs[k].first].name}, self._cmodel.collisionPairs[k].second : {self._cmodel.collisionPairs[k].second}")
            # colres = ResidualCollision(
            #     self._state,
            #     self._cmodel,
            #     self._cdata,
            #     k,
            #     self._cmodel.geometryObjects[
            #         self._cmodel.collisionPairs[k].first
            #     ].parentJoint,
            # )
            # constraint = crocoddyl.ConstraintModelResidual(
            #     self._state,
            #     colres,
            #     np.array([0.0]),
            #     np.array([np.inf]),
            # )
            
            # self._runningConstraintModelManager.addConstraint("col", constraint)
            # self._terminalConstraintModelManager.addConstraint("col_term", constraint)
            
            
            colcost = CostModelPairCollision(
                self._state,
                self._cmodel,
                self._cdata,
                k,
                self._cmodel.geometryObjects[
                    self._cmodel.collisionPairs[k].first
                ].parentJoint,
            )
            self._runningCostModel.addCost(
                "col" + str(k), colcost, self._WEIGHT_COL
            )
            self._terminalCostModel.addCost(
                "col_term_" + str(k), colcost, self._WEIGHT_COL
            )
       # Adding costs to the models
        self._runningCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
        self._runningCostModel.addCost("ctrlRegGrav", uRegCost, self._WEIGHT_uREG)
        self._runningCostModel.addCost("gripperPoseRM", goalTrackingCost, self._WEIGHT_GRIPPER_POSE)        
        self._terminalCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
        self._terminalCostModel.addCost(
            "gripperPose", goalTrackingCost, self._WEIGHT_GRIPPER_POSE
        )
         # Bounds costs
        
                # Cost for self-collision
        maxfloat = sys.float_info.max
        xlb = np.concatenate(
            [
                self._rmodel.lowerPositionLimit,
                -maxfloat * np.ones(self._state.nv),
            ]
        )
        xub = np.concatenate(
            [
                self._rmodel.upperPositionLimit,
                maxfloat * np.ones(self._state.nv),
            ]
        )
        bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(self._state, self._x0, self._actuation.nu)
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        limitCost = crocoddyl.CostModelResidual(self._state, xLimitActivation, xLimitResidual)
        
        self._runningCostModel.addCost("limitCost", limitCost, self._WEIGHT_LIMIT)  
        self._terminalCostModel.addCost("limitCost", limitCost, self._WEIGHT_LIMIT)

        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        self._running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(self._state, self._actuation, self._runningCostModel, self._runningConstraintModelManager)
        self._terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(self._state, self._actuation, self._terminalCostModel, self._terminalConstraintModelManager)

        # self._running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(self._state, self._actuation, self._runningCostModel)
        # self._terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(self._state, self._actuation, self._terminalCostModel)
        
                
        # self._running_DAM = crocoddyl.DifferentialActionModelNumDiff(
        #     self._running_DAM, True
        # )
       
        # self._terminal_DAM = crocoddyl.DifferentialActionModelNumDiff(
        #     self._terminal_DAM, True
        # )
        
        # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
        self._runningModel = crocoddyl.IntegratedActionModelEuler(self._running_DAM, self._dt)
        self._terminalModel = crocoddyl.IntegratedActionModelEuler(self._terminal_DAM, 0.)
        
        
        # Optionally add armature to take into account actuator's inertia
        self._runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
        self._terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
        
        problem = crocoddyl.ShootingProblem(self._x0, [self._runningModel] * self._T, self._terminalModel)
        # Create solver + callbacks
        # ddp = crocoddyl.SolverFDDP(problem)
        
        # ddp.setCallbacks([crocoddyl.CallbackLogger(),
        #                 crocoddyl.CallbackVerbose()])
        
        
        # Define solver
        ddp = mim_solvers.SolverSQP(problem)
        # ddp = mim_solvers.SolverSQP(problem)
        ddp.use_filter_line_search = True
        ddp.termination_tolerance = 1e-10
        ddp.max_qp_iters = 10000
        ddp.with_callbacks = True 
        
        return ddp