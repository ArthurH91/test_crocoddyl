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
        if data.d <=  0:
            data.residual.r[:] = self.res.w
        else:
            data.residual.r[:].fill(0.0)
        data.cost = 1 / 2 * data.d**2

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
            J = (jacobian.T @ self.res_diff.dn_loc_dM1.T).T

            # compute the residual derivatives
            data.residual.Rx[:3, :nv] = J[:3, :]
        else:
            data.residual.Rx[:3, :nv].fill(0.0)

    def createData(self, collector):
        data = CostDataPairCollision(self, collector)
        return data


class CostDataPairCollision(crocoddyl.CostDataAbstract):
    def __init__(self, model, data_collector):
        crocoddyl.CostDataAbstract.__init__(self, model, data_collector)
        self.d = np.zeros(3)


if __name__ == "__main__":
    pass
