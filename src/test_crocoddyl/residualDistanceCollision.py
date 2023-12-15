import numpy as np
import pinocchio as pin
import crocoddyl
from crocoddyl.utils import *
import pydiffcol

from utils import select_strategy


class ResidualCollision(crocoddyl.ResidualModelAbstract):
    """Class computing the residual of the collision constraint. This residual is simply the signed distance between the two closest points of the 2 shapes."""

    def __init__(
        self,
        state,
        geom_model: pin.Model,
        geom_data,
        pair_id: int,
    ):
        """Class computing the residual of the collision constraint. This residual is simply the signed distance between the two closest points of the 2 shapes.

        Args:
            state (crocoddyl.StateMultibody): _description_
            geom_model (pin.Model): Collision model of pinocchio
            geom_data (_type_): Collision data of the collision model of pinocchio
            pair_id (int): ID of the collision pair
        """
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, True, True, True)

        # Pinocchio robot model
        self._pinocchio = self.state.pinocchio

        # Geometry model of the robot
        self._geom_model = geom_model

        # Geometry data of the robot
        self._geom_data = geom_data

        # Pair ID of the collisionPair
        self.pair_id = pair_id

        # Number of joints
        self.nq = self._pinocchio.nq

        # Making sure that the pair of collision exists
        assert self.pair_id <= len(self._geom_model.collisionPairs)

        # Collision pair
        self._collisionPair = self._geom_model.collisionPairs[0]

        # Geometry ID of the shape 1 of collision pair
        self._shape1_id = self._collisionPair.first

        # Making sure that the frame exists
        assert self._shape1_id <= len(self._geom_model.geometryObjects)

        # Geometry object shape 1
        self._shape1 = self._geom_model.geometryObjects[self._shape1_id]

        # Shape 1 parent joint
        self._shape1_parentJoint = self._shape1.parentJoint

        # Geometry ID of the shape 2 of collision pair
        self._shape2_id = self._collisionPair.second

        # Making sure that the frame exists
        assert self._shape2_id <= len(self._geom_model.geometryObjects)

        # Geometry object shape 2
        self._shape2 = self._geom_model.geometryObjects[self._shape2_id]

        # Shape 2 parent joint
        self._shape2_parentJoint = self._shape2.parentJoint

        # Checking that shape 1 is belonging to the robot & shape 2 is the obstacle
        assert not "obstacle" in self._shape1.name
        assert "obstacle" in self._shape2.name

    def calc(self, data, x, u=None):
        data.r[:] = self.f(data, x[: self.nq])

    def f(self, data, q):
        # Storing q outside of the state vector
        self.q = q

        ### Computes the distance for the collision pair pair_id
        # Updating the position of the joints & the geometry objects.
        pin.updateGeometryPlacements(
            self._pinocchio,
            data.shared.pinocchio,
            self._geom_model,
            self._geom_data,
            self.q,
        )

        # Distance Request & Result from hppfcl / pydiffcol
        self._req, self._req_diff = select_strategy("first_order_gaussian")
        self._res = pydiffcol.DistanceResult()
        self._res_diff = pydiffcol.DerivativeResult()

        # Getting the geometry of the shape 1
        self._shape1_geom = self._shape1.geometry

        # Getting its pose in the world reference
        self._shape1_placement = self._geom_data.oMg[self._shape1_id]

        # Doing the same for the second shape.
        self._shape2_geom = self._shape2.geometry
        self._shape2_placement = self._geom_data.oMg[self._shape2_id]

        # Computing the distance
        distance = pydiffcol.distance(
            self._shape1_geom,
            self._shape1_placement,
            self._shape2_geom,
            self._shape2_placement,
            self._req,
            self._res,
        )
        
        dist2 = np.linalg.norm(self._shape1_placement.translation - self._shape2_placement.translation)
        print(f"dist2 = {dist2}")
        print(f"distance : {distance}")
        return distance

    def calcDiff(self, data, x, u = None):
        self.derivative_diffcol(data, x, u = None)
        print(f"self._J : {self._J}")

        self.calcDiff_numdiff(data, x)
        print(f"self._J numdiff: {self._J}")
        
        print("__________________")
        data.Rx[:self.nq] = self._J
        
    def derivative_diffcol(self, data, x, u=None):
    # Storing nq outside of state.
        nq = self.state.nq

        # Storing q outside of the state vector.
        self.q = np.array(x[:nq])

        ### Computes the distance for the collision pair pair_id
        # Updating the position of the joints & the geometry objects.
        pin.updateGeometryPlacements(
            self._pinocchio,
            data.shared.pinocchio,
            self._geom_model,
            self._geom_data,
            self.q,
        )
        pin.forwardKinematics(self._pinocchio,  data.shared.pinocchio,  self.q)
        pin.updateGeometryPlacements(self._pinocchio,  data.shared.pinocchio, self._geom_model, self._geom_data,  self.q)

        pin.computeJointJacobians(self._pinocchio,  data.shared.pinocchio, self.q )
    # Computing the pinocchio jacobians
        pin.computeJointJacobians(self._pinocchio, data.shared.pinocchio, self.q)

        # Computing the distance derivatives of pydiffcol
        pydiffcol.distance_derivatives(
            self._shape1_geom,
            self._shape1_placement,
            self._shape2_geom,
            self._shape2_placement,
            self._req,
            self._res,
            self._req_diff,
            self._res_diff,
        )

        # Jacobian of the parent frame of the shape1
        jacobian = pin.computeFrameJacobian(
            self._pinocchio,
            data.shared.pinocchio,
            self.q,
            self._shape1.parentFrame,
            pin.LOCAL,
        )

        # The jacobian here is the multiplication of the jacobian of the end effector and the jacobian of the distance between the geometry object and the obstacle
        self._J = np.dot(self._res_diff.ddist_dM1, jacobian)
        # print(f"self.calcDiff_numdiff(data, x) : {self.calcDiff_numdiff(data, x) }")
        # print(f"J : {J}")
        # assert np.isclose(np.linalg.norm(self.calcDiff_numdiff(data, x)), np.linalg.norm(J), 1e-3) 
        # compute the residual derivatives
        data.Rx[:nq] = self._J
    
    def calcDiff_numdiff(self, data,x):
        j_diff = np.zeros(self.nq)
        fx = self.f(data, x[: self.nq])
        for i in range(self.nq):
            e = np.zeros(self.nq)
            e[i] = 1e-6
            j_diff[i] = (self.f(data, x[: self.nq] + e) - fx) / e[i]
        self._J = j_diff


if __name__ == "__main__":
    pass
