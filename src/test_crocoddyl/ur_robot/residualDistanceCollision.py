import numpy as np
import pinocchio as pin
import crocoddyl
from crocoddyl.utils import *
import hppfcl



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
        self._pair_id = pair_id

        # Number of joints
        self._nq = self._pinocchio.nq

        # Making sure that the pair of collision exists
        assert self._pair_id <= len(self._geom_model.collisionPairs)

        # Collision pair
        self._collisionPair = self._geom_model.collisionPairs[self._pair_id]

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
        data.r[:] = self.f(data, x[: self._nq])

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
        self._req = hppfcl.DistanceRequest()
        self._res = hppfcl.DistanceResult()

        # Getting the geometry of the shape 1
        self._shape1_geom = self._shape1.geometry

        # Getting its pose in the world reference
        self._shape1_placement = self._geom_data.oMg[self._shape1_id]

        # Doing the same for the second shape.
        self._shape2_geom = self._shape2.geometry
        self._shape2_placement = self._geom_data.oMg[self._shape2_id]

        # Computing the distance
        distance = hppfcl.distance(
            self._shape1_geom,
            hppfcl.Transform3f(self._shape1_placement.rotation, self._shape1_placement.translation),
            self._shape2_geom,
            hppfcl.Transform3f(self._shape2_placement.rotation, self._shape2_placement.translation),
            self._req,
            self._res,
        )
        
        return distance

    def calcDiff(self, data, x, u=None):
        # Computing the distance
        # distance = pydiffcol.distance(
        #     self._shape1_geom,
        #     self._shape1_placement,
        #     self._shape2_geom,
        #     self._shape2_placement,
        #     self._req,
        #     self._res,
        # )

        # dist2 = np.linalg.norm(self._shape1_placement.translation - self._shape2_placement.translation) - 1.5e-1 - 0.055
        # print(f"dist diff = {dist2 - distance}")
        # # print(f"distance : {distance}")
        # print(f"self._shape1_placement : {self._shape1_placement}")
        # print(f"self._shape2_placement : {self._shape2_placement}")
        # print(f"self._shape1_radius : {self._shape1_geom.radii}")
        # print(f"self._shape2_radius : {self._shape2_geom.radius}")

        # self.derivative_diffcol(data, x, u = None)
        # print(f"self._J : {self._J}")

        # self.calcDiff_numdiff(data, x)
        # J_n = self._J
        
        self.calcDiff_florent(data,x)
        J_f = self._J
    
        # print((np.isclose(J_n, J_f, rtol=1e-4)))
        # if False in np.isclose(J_n, J_f, rtol=1e-3):
        #     print(x[:6].tolist())
        #     print(f"dist_numdiff = {self._distance_numdiff}")
        #     print(f"dist_florent = {self._distance_florent}")
        # # assert False in np.isclose(J_n, J_f, rtol=1e-3)
        #     print(f"self._J numdiff: {J_n}")
        #     print(f"self._J florent: {J_f}")

        # print(f"q : {x[:self._nq]}")

        # print("__________________")
        data.Rx[: self._nq] = self._J



    def calcDiff_numdiff(self, data, x):
        j_diff = np.zeros(self._nq)
        fx = self.f(data, x[: self._nq])
        for i in range(self._nq):
            e = np.zeros(self._nq)
            e[i] = 1e-6
            j_diff[i] = (self.f(data, x[: self._nq] + e) - fx) / e[i]
        self._J = j_diff
        self._distance_numdiff = fx

    def calcDiff_florent(self, data, x):
        
        # pin.forwardKinematics(self._pinocchio, data.shared.pinocchio, self.q)
        # pin.updateGeometryPlacements(
        #     self._pinocchio,
        #     data.shared.pinocchio,
        #     self._geom_model,
        #     self._geom_data,
        #     self.q,
        # )
        jacobian = pin.computeFrameJacobian(
            self._pinocchio,
            data.shared.pinocchio,
            self.q,
            self._shape1.parentFrame,
            pin.LOCAL_WORLD_ALIGNED,
        )

        # Computing the distance
        self._distance_florent = self.f(data, x[: self._nq])
        sign = 1
        if self._distance_florent < 0:
            sign = - 1 
        cp1 = self._res.getNearestPoint1()
        cp2 = self._res.getNearestPoint2()

        self._J = sign * (cp1 - cp2).T / self._distance_florent @ jacobian[:3]



if __name__ == "__main__":
    pass
