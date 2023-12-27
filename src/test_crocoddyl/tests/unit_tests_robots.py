import unittest

from os.path import dirname, join, abspath
import numpy as np

import example_robot_data as robex
import pinocchio as pin
import hppfcl


class TestRobotsDistanceDerivatives(unittest.TestCase):
    """This class is made to test the distances derivatives between primitives pairs such as sphere-sphere of panda robot & ur10. The collisions shapes are from hppfcl."""

    def load_panda(self):
        """Load the robot from the models folder.

        Returns:
            rmodel, vmodel, cmodel: Robot model, visual model & collision model of the robot.
        """

        ### LOADING THE ROBOT
        pinocchio_model_dir = join(
            dirname(dirname(dirname(str(abspath(__file__))))), "models"
        )
        model_path = join(pinocchio_model_dir, "franka_description/robots")
        mesh_dir = pinocchio_model_dir
        urdf_filename = "franka2.urdf"
        urdf_model_path = join(join(model_path, "panda"), urdf_filename)

        rmodel, cmodel, vmodel = pin.buildModelsFromUrdf(
            urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
        )

        q0 = pin.neutral(rmodel)

        rmodel, [vmodel, cmodel] = pin.buildReducedModel(
            rmodel, [vmodel, cmodel], [1, 9, 10], q0
        )

        ### CREATING THE SPHERE ON THE UNIVERSE
        SPHERE1_RADIUS = 1.5e-1
        SPHERE1_POSE = pin.SE3.Identity()
        SPHERE1_POSE.translation = np.array([0.0, 0.25, 1.5])
        SPHERE1 = hppfcl.Sphere(SPHERE1_RADIUS)
        SPHERE1_GEOM_OBJECT = pin.GeometryObject(
            "SPHERE1",
            rmodel.getFrameId("universe"),
            rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
            SPHERE1,
            SPHERE1_POSE,
        )
        self.ID_SPHERE1_PA = cmodel.addGeometryObject(SPHERE1_GEOM_OBJECT)

        ### CREATING THE SPHERE ON THE END EFFECTOR
        SPHERE2_RADIUS = 1.5e-1
        SPHERE2_POSE = pin.SE3.Identity()
        SPHERE2_POSE.translation = np.array([0.2, 0.0, 0.0])
        SPHERE2 = hppfcl.Sphere(SPHERE2_RADIUS)
        SPHERE2_GEOM_OBJECT = pin.GeometryObject(
            "SPHERE2",
            rmodel.getFrameId("panda2_leftfinger"),
            rmodel.frames[rmodel.getFrameId("panda2_leftfinger")].parentJoint,
            SPHERE2,
            SPHERE2_POSE,
        )
        self.ID_SPHERE2_PA = cmodel.addGeometryObject(SPHERE2_GEOM_OBJECT)

        ### CREATING THE SPHERE ON THE ROBOT
        SPHERE3_RADIUS = 1.5e-1
        SPHERE3_POSE = pin.SE3.Identity()
        SPHERE3_POSE.translation = np.array([0.0, 0.1, 0.2])
        SPHERE3 = hppfcl.Sphere(SPHERE3_RADIUS)
        SPHERE3_GEOM_OBJECT = pin.GeometryObject(
            "SPHERE3",
            rmodel.getFrameId("panda2_link3_sc_joint"),
            rmodel.frames[rmodel.getFrameId("panda2_link3_sc_joint")].parentJoint,
            SPHERE3,
            SPHERE3_POSE,
        )
        self.ID_SPHERE3_PA = cmodel.addGeometryObject(SPHERE3_GEOM_OBJECT)

        return rmodel, vmodel, cmodel

    def load_ur(self):
        robot = robex.load("ur10")
        rmodel = robot.model
        cmodel = robot.collision_model
        vmodel = robot.visual_model

        ### CREATING THE SPHERE ON THE UNIVERSE
        SPHERE1_RADIUS = 1.5e-1
        SPHERE1_POSE = pin.SE3.Identity()
        SPHERE1_POSE.translation = np.array([0.0, 0.25, 1.5])
        SPHERE1 = hppfcl.Sphere(SPHERE1_RADIUS)
        self.SPHERE1_GEOM_OBJECT = pin.GeometryObject(
            "SPHERE1",
            rmodel.getFrameId("universe"),
            rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
            SPHERE1,
            SPHERE1_POSE,
        )
        self.ID_SPHERE1_UR = cmodel.addGeometryObject(self.SPHERE1_GEOM_OBJECT)

        ### CREATING THE SPHERE ON THE END EFFECTOR
        SPHERE2_RADIUS = 1.5e-1
        SPHERE2_POSE = pin.SE3.Identity()
        SPHERE2_POSE.translation = np.array([0.2, 0.0, 0.0])
        SPHERE2 = hppfcl.Sphere(SPHERE2_RADIUS)
        self.SPHERE2_GEOM_OBJECT = pin.GeometryObject(
            "SPHERE2",
            rmodel.getFrameId("tool0"),
            rmodel.frames[rmodel.getFrameId("tool0")].parentJoint,
            SPHERE2,
            SPHERE2_POSE,
        )
        self.ID_SPHERE2_UR = cmodel.addGeometryObject(self.SPHERE2_GEOM_OBJECT)

        ### CREATING THE SPHERE ON THE ROBOT
        SPHERE3_RADIUS = 1.5e-1
        SPHERE3_POSE = pin.SE3.Identity()
        SPHERE3_POSE.translation = np.array([0.0, 0.3, 0.0])
        SPHERE3 = hppfcl.Sphere(SPHERE3_RADIUS)
        self.SPHERE3_GEOM_OBJECT = pin.GeometryObject(
            "SPHERE3",
            rmodel.getFrameId("wrist_2_joint"),
            rmodel.frames[rmodel.getFrameId("wrist_2_joint")].parentJoint,
            SPHERE3,
            SPHERE3_POSE,
        )
        self.ID_SPHERE3_UR = cmodel.addGeometryObject(self.SPHERE3_GEOM_OBJECT)

        return rmodel, vmodel, cmodel

    def test_distance(self):
        # Loading the UR robot
        rmodel_ur, vmodel_ur, cmodel_ur = self.load_ur()

        rdata_ur = rmodel_ur.createData()
        cdata_ur = cmodel_ur.createData()

        q0_ur = pin.neutral(rmodel_ur)

        # Loading the panda robot 
        rmodel_pa, vmodel_pa, cmodel_pa = self.load_panda()

        rdata_pa = rmodel_pa.createData()
        cdata_pa = cmodel_pa.createData()

        q0_pa = pin.neutral(rmodel_pa)        
        
        # Number of joints
        nq_ur = rmodel_ur.nq
        nq_pa = rmodel_pa.nq

        # Updating the models
        pin.forwardKinematics(rmodel_ur, rdata_ur, q0_ur)
        pin.framesForwardKinematics(rmodel_ur, rdata_ur, q0_ur)
        pin.updateGeometryPlacements(rmodel_ur, rdata_ur, cmodel_ur, cdata_ur, q0_ur)

        pin.forwardKinematics(rmodel_pa, rdata_pa, q0_pa)
        pin.framesForwardKinematics(rmodel_pa, rdata_pa, q0_pa)
        pin.updateGeometryPlacements(rmodel_pa, rdata_pa, cmodel_pa, cdata_pa, q0_pa)

        # Distance & Derivative results from hppfcl
        req = hppfcl.DistanceRequest()
        res = hppfcl.DistanceResult()

        ### Distance between sphere 1 (on universe) & sphere 2 (on tool0)

        distance_ur = self.dist(rmodel_ur, rdata_ur, cmodel_ur, cdata_ur, self.ID_SPHERE1_UR, self.ID_SPHERE2_UR, res, req, q0_ur)
        distance_pa = self.dist(rmodel_pa, rdata_pa, cmodel_pa, cdata_pa, self.ID_SPHERE1_PA, self.ID_SPHERE2_PA, res, req, q0_pa)

        self.assertAlmostEqual(distance_ur,1.4865013858018128)
        self.assertAlmostEqual(distance_pa,0.28841673157720465)


    def dist(self, rmodel, rdata, cmodel, cdata, shape1_id, shape2_id, res, req, q):
        """Computes the distance with diffcol

        Args:
            q (np.ndarray): Configuration of the robot

        Returns:
            distance : distance between shape 1 & shape 2
        """
        # Computing the distance
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

        shape1_placement = cdata.oMg[shape1_id]
        shape2_placement = cdata.oMg[shape2_id]

        distance = hppfcl.distance(
            cmodel.geometryObjects[shape1_id].geometry,
            shape1_placement,
            cmodel.geometryObjects[shape2_id].geometry,
            shape2_placement,
            req,
            res,
        )
        return distance


if __name__ == "__main__":
    unittest.main()
