from copy import copy
import numpy as np
import pinocchio as pin
import crocoddyl
from crocoddyl.utils import *
import pydiffcol
from utils import select_strategy


class CostModelPairCollision(crocoddyl.CostModelAbstract):
    def __init__(
        self,
        state,
        geom_model: pin.Model,
        geom_data,
        pair_id: int,
        joint_id: int,
        parentFrame: int,
    ):
        """_summary_

        Args:
            state (crocoddyl.StateMultibody): _description_
            geom_model (pin.CollisionModel): _description_
            pair_id (tuple): _description_
            joint_id (int): _description_
        """
        r_activation = 3

        crocoddyl.CostModelAbstract.__init__(
            self, state, crocoddyl.ActivationModelQuad(r_activation)
        )

        self.pinocchio = self.state.pinocchio
        self.geom_model = geom_model
        self.pair_id = pair_id
        self.joint_id = joint_id
        self.geom_data = geom_data
        self.parentFrame = parentFrame

    def calc(self, data, x, u=None):
        self.q = np.array(x[: self.state.nq])
        # computes the distance for the collision pair pair_id
        pin.forwardKinematics(self.pinocchio, data.shared.pinocchio, self.q)
        pin.updateGeometryPlacements(
            self.pinocchio,
            data.shared.pinocchio,
            self.geom_model,
            self.geom_data,
            self.q,
        )

        self.req, self.req_diff = select_strategy("first_order_gaussian")
        self.res = pydiffcol.DistanceResult()
        self.res_diff = pydiffcol.DerivativeResult()
        self.shape1_id = self.geom_model.collisionPairs[self.pair_id].first
        self.shape1_geom = self.geom_model.geometryObjects[self.shape1_id].geometry
        self.shape1_placement = self.geom_data.oMg[self.shape1_id]

        self.shape2_id = self.geom_model.collisionPairs[self.pair_id].second
        self.shape2_geom = self.geom_model.geometryObjects[self.shape2_id].geometry
        self.shape2_placement = self.geom_data.oMg[self.shape2_id]

        data.d = pydiffcol.distance(
            self.shape1_geom,
            self.shape1_placement,
            self.shape2_geom,
            self.shape2_placement,
            self.req,
            self.res,
        )
        
        # calculate residual
        if self.res.min_distance <= 0:
            data.residual.r[:] = self.res.w
            data.cost = 0.5 * np.dot(self.res.w, self.res.w)

        else:
            data.residual.r[:].fill(0.0)
            data.cost= 0
        
        self.res_numdiff = data.residual.r[:]
            
    def calcDiff(self, data, x, u=None):
        nv = self.state.nv

        if self.res.min_distance <= 0:
            pin.computeJointJacobians(self.pinocchio, data.shared.pinocchio, self.q)

            pydiffcol.distance_derivatives(
                self.shape1_geom,
                self.shape1_placement,
                self.shape2_geom,
                self.shape2_placement,
                self.req,
                self.res,
                self.req_diff,
                self.res_diff,
            )

            jacobian = pin.computeFrameJacobian(
                self.pinocchio,
                data.shared.pinocchio,
                self.q,
                self.parentFrame,
                pin.LOCAL,
            )

            # The jacobian here is the multiplication of the jacobian of the end effector and the jacobian of the distance between the geometry object and the obstacle
            J = np.dot(self.res_diff.dw_dM1, jacobian)

            # compute the residual derivatives
            data.residual.Rx[:3, :self.state.nq] = J
            
        else:
            data.residual.Rx[:3, :nv].fill(0.0)
        
        data.Lx[:] = np.dot(data.residual.Rx.T, data.residual.r)
        data.Lxx[:] = np.dot(data.residual.Rx.T, data.residual.Rx)
        
    def numdiff(self, f, x, data, eps=1e-8):
        """Estimate df/dx at x with finite diff of step eps

        Parameters
        ----------
        f : function handle
            Function evaluated for the finite differente of its gradient.
        x : np.ndarray
            Array at which the finite difference is calculated
        eps : float, optional
            Finite difference step, by default 1e-6

        Returns
        -------
        jacobian : np.ndarray
            Finite difference of the function f at x.
        """
        xc = np.copy(x)
        f(data, x)
        f0 = np.copy(self.res_numdiff)
        res = []
        for i in range(self.state.nq):
            xc[i] += eps
            f(data, xc)
            fc = self.res_numdiff
            res.append(copy(fc - f0) / eps)
            xc[i] = x[i]
        return np.array(res).T

           
    def createData(self, collector):
        data = CostDataPairCollision(self, collector)
        return data


class CostDataPairCollision(crocoddyl.CostDataAbstract):
    def __init__(self, model, data_collector):
        crocoddyl.CostDataAbstract.__init__(self, model, data_collector)
        self.d = np.zeros(3)


if __name__ == "__main__":
    pass