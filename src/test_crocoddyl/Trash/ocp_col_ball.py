## Class heavily inspired by the work of Sebastien Kleff : https://github.com/machines-in-motion/minimal_examples_crocoddyl

from typing import Any
import numpy as np
import crocoddyl
import pinocchio as pin
import mim_solvers


from residualCollision import ResidualCollision


class OCPPandaReachingCol:
    """This class is creating a optimal control problem of a panda robot reaching for a target while taking auto collisions into consideration"""

    def __init__(
        self,
        rmodel: pin.Model,
        cmodel: pin.GeometryModel,
        cdata,
        TARGET_POSE: pin.SE3,
        OBSTACLE_POSE: pin.SE3,
        T: int,
        dt: float,
        x0: np.ndarray,
        WEIGHT_XREG=1e-1,
        WEIGHT_UREG=1e-2,
        WEIGHT_COL=1e0,
        WEIGHT_TERM_COL=1e-2,
        WEIGHT_TERM_POS=10,
        RUNNING_COST_ENDEFF = True,
    ) -> None:
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
        self._cdata = cdata
        self._TARGET_POSE = TARGET_POSE
        self._OBSTACLE_POSE = OBSTACLE_POSE
        self._T = T
        self._dt = dt
        self._x0 = x0

        # Weights
        self._WEIGHT_xREG = WEIGHT_XREG
        self._WEIGHT_uREG = WEIGHT_UREG
        self._WEIGHT_COL = WEIGHT_COL
        self._WEIGHT_TERM_COL = WEIGHT_TERM_COL
        self._WEIGHT_TERM_POS = WEIGHT_TERM_POS
        
        # Options
        self._RUNNING_COST_ENDEFF = RUNNING_COST_ENDEFF

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
            self._state,
            self._rmodel.getFrameId("panda2_leftfinger"),
            self._TARGET_POSE.translation,
        )
        
        goalTrackingCost = crocoddyl.CostModelResidual(
            self._state, framePlacementResidual
        )
        self._runningConstraintModelManager = crocoddyl.ConstraintModelManager(self._state, self._actuation.nu)
        self._terminalConstraintModelManager = crocoddyl.ConstraintModelManager(self._state, self._actuation.nu)




        # # Collision residuals
        # for k in range(len(self._cmodel.collisionPairs)):
        #     colres = crocoddyl.ResidualModelFrameTranslation(
        #         self._state,
        #         self._cmodel.geometryObjects[self._cmodel.collisionPairs[k].first].parentFrame,
        #         self._OBSTACLE_POSE.translation
        #     )
        #     print(f"obs pose : {self._OBSTACLE_POSE.translation}")
        #     constraint = crocoddyl.ConstraintModelResidual(
        #         self._state,
        #         colres,
        #         np.array([5e-2,5e-2,5e-2]),
        #         np.array([np.inf, np.inf,np.inf]),
        #     )
            
        #     self._runningConstraintModelManager.addConstraint("col" + str(k), constraint)
        #     self._terminalConstraintModelManager.addConstraint("col_term_" + str(k), constraint)


        # Adding costs to the models
        self._runningCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
        self._runningCostModel.addCost("ctrlRegGrav", uRegCost, self._WEIGHT_uREG)
        if self._RUNNING_COST_ENDEFF:
            self._runningCostModel.addCost(
                "gripperPoseRM", goalTrackingCost, self._WEIGHT_TERM_POS
            )
        self._terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
        self._terminalCostModel.addCost(
            "gripperPose", goalTrackingCost, self._WEIGHT_TERM_POS
        )

        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        self._init_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state, self._actuation, self._runningCostModel
        )
        self._running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state, self._actuation, self._runningCostModel, self._runningConstraintModelManager
        )
        self._terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state, self._actuation, self._terminalCostModel, self._terminalConstraintModelManager
        )

        # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
         
        self._initModel = crocoddyl.IntegratedActionModelEuler(
            self._init_DAM, self._dt
        )
        
        self._running_DAM = crocoddyl.DifferentialActionModelNumDiff(
            self._running_DAM, True
        )
       
        self._terminal_DAM = crocoddyl.DifferentialActionModelNumDiff(
            self._terminal_DAM, True
        )
        
        self._runningModel = crocoddyl.IntegratedActionModelEuler(
            self._running_DAM, self._dt
        )
       
        self._terminalModel = crocoddyl.IntegratedActionModelEuler(
            self._terminal_DAM, self._dt
        )

        problem = crocoddyl.ShootingProblem(
            self._x0, [self._initModel] + [self._runningModel] *( self._T - 1), self._terminalModel
        )
        # Create solver + callbacks
        ddp = crocoddyl.SolverFDDP(problem)

        ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# Define solver
        # ddp = mim_solvers.SolverSQP(problem)
        # ddp.use_filter_line_search = False
        # ddp.termination_tolerance = 1e-4
        # # ddp.max_qp_iters = 10000
        # ddp.with_callbacks = True 
        return ddp
