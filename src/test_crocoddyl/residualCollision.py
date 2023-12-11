from copy import copy
import numpy as np
import pinocchio as pin
import crocoddyl
from crocoddyl.utils import *
import pydiffcol
from utils import select_strategy


class ResidualCollision(crocoddyl.ResidualModelAbstract):
    def __init__(
                self,
        state,
        geom_model: pin.Model,
        geom_data,
        pair_id: int,
        joint_id: int,
    ):
        
        crocoddyl.ResidualModelAbstract.__init__(self, state, 3, True, True, True)
        
        
        self.pinocchio = self.state.pinocchio
        self.geom_model = geom_model
        self.pair_id = pair_id
        self.joint_id = joint_id
        self.geom_data = geom_data
        self._eps = 5e-3


    def calc(self, data, x, u=None):
        
        
                # Storing q outside of the state vector
        self.q = np.array(x[: self.state.nq])

        ### Computes the distance for the collision pair pair_id
        # Updating the position of the joints & the geometry objects.
        pin.updateGeometryPlacements(
            self.pinocchio,
            data.shared.pinocchio,
            self.geom_model,
            self.geom_data,
            self.q,
        )

        # Distance Request & Result from hppfcl / pydiffcol
        self.req, self.req_diff = select_strategy("first_order_gaussian")
        self.res = pydiffcol.DistanceResult()
        self.res_diff = pydiffcol.DerivativeResult()

        # Getting the ID of the first shape from the collision pair id
        self.shape1_id = self.geom_model.collisionPairs[self.pair_id].first

        # Getting its geometry
        self.shape1_geom = self.geom_model.geometryObjects[self.shape1_id].geometry

        # Getting its pose
        self.shape1_placement = self.geom_data.oMg[self.shape1_id]

        # Parent frame
        self.parentFrame = self.geom_model.geometryObjects[self.shape1_id].parentFrame
        # Doing the same for the second shape.
        self.shape2_id = self.geom_model.collisionPairs[self.pair_id].second
        self.shape2_geom = self.geom_model.geometryObjects[self.shape2_id].geometry
        self.shape2_placement = self.geom_data.oMg[self.shape2_id]

        # Computing the distance
        data.d = pydiffcol.distance(
            self.shape1_geom,
            self.shape1_placement,
            self.shape2_geom,
            self.shape2_placement,
            self.req,
            self.res,
        )
        # data.r[:] = self.res.w - 2 * self._eps * self.res.n
        data.r[:] = self.res.w

    def calcDiff(self, data, x, u=None):
        pass
# Storing nq outside of state.
#         nq = self.state.nq

#         # Storing q outside of the state vector.
#         self.q = np.array(x[:nq])

#         ### Computes the distance for the collision pair pair_id
#         # Updating the position of the joints & the geometry objects.
#         pin.updateGeometryPlacements(
#             self.pinocchio,
#             data.shared.pinocchio,
#             self.geom_model,
#             self.geom_data,
#             self.q,
#         )

# # Computing the pinocchio jacobians
#         pin.computeJointJacobians(self.pinocchio, data.shared.pinocchio, self.q)

#         # Computing the distance derivatives of pydiffcol
#         pydiffcol.distance_derivatives(
#             self.shape1_geom,
#             self.shape1_placement,
#             self.shape2_geom,
#             self.shape2_placement,
#             self.req,
#             self.res,
#             self.req_diff,
#             self.res_diff,
#         )

#         # Jacobian of the parent frame of the shape1
#         jacobian = pin.computeFrameJacobian(
#             self.pinocchio,
#             data.shared.pinocchio,
#             self.q,
#             self.parentFrame,
#             pin.LOCAL,
#         )

#         # The jacobian here is the multiplication of the jacobian of the end effector and the jacobian of the distance between the geometry object and the obstacle
#         J = np.dot(self.res_diff.dw1_loc_dM1, jacobian) 
#         # compute the residual derivatives
#         data.Rx[:3, :nq] = J



# if __name__ == "__main__":
#     pass
