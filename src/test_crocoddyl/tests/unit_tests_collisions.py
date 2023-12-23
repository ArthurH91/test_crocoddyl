import unittest

import numpy as np

import pinocchio as pin
import hppfcl


class TestCollisions(unittest.TestCase):
    """This class is made to test the collisions between primitives pairs such as sphere-sphere. The collisions shapes are from hppfcl."""

    def test_sphere_sphere_not_in_collision(self):
        """Testing the sphere-sphere pair, going from the distance between each shape to making sure the closest points are well computed."""


        r1 = 0.4
        r2 = 0.5
        
        rmodel = pin.Model()
        cmodel = pin.GeometryModel()
        geometries = [
            hppfcl.Sphere(r1),
            hppfcl.Sphere(r2),
        ]
        # With pinocchio3, a new way of constructing a geometry object is available and the old one will be deprecated.
        for i, geom in enumerate(geometries):
            placement = pin.SE3(np.eye(3), np.array([i, 0, 0]))
            try:
                geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, placement, geom)
            except:
                geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, geom, placement)
            cmodel.addGeometryObject(geom_obj)

        rdata = rmodel.createData()
        cdata = cmodel.createData()

        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        # Creating the shapes for the collision detection.
        shape1_id = cmodel.getGeometryId("obj0")

        # Coloring the sphere
        shape1 = cmodel.geometryObjects[shape1_id]
        self.assertIsInstance(shape1.geometry, hppfcl.Sphere)

        # Getting its pose in the world reference
        shape1_placement = cdata.oMg[shape1_id]

        # Doing the same for the second shape.
        shape2_id = cmodel.getGeometryId("obj1")

        # Coloring the sphere
        shape2 = cmodel.geometryObjects[shape2_id]
        self.assertIsInstance(shape2.geometry, hppfcl.Sphere)

        # Getting its pose in the world reference
        shape2_placement = cdata.oMg[shape2_id]

        req = hppfcl.DistanceRequest()
        res = hppfcl.DistanceResult()

        # Testing the distance calculus
        distance_hpp = hppfcl.distance(
            shape1.geometry,
            hppfcl.Transform3f(shape1_placement.rotation, shape1_placement.translation),
            shape2.geometry,
            hppfcl.Transform3f(shape2_placement.rotation, shape2_placement.translation),
            req,
            res,
        )

        distance_ana = np.linalg.norm(shape1_placement.translation - shape2_placement.translation) - (r1 + r2)
        self.assertAlmostEqual(distance_ana, distance_hpp)
        
        # Testing the computation of closest points
        cp1 = res.getNearestPoint1()
        cp2 = res.getNearestPoint2()
        
        self.assertAlmostEqual(np.linalg.norm(cp1 - cp2), distance_ana)
        
        
    def test_sphere_sphere_in_collision(self):
        """Testing the sphere-sphere pair, going from the distance between each shape to making sure the closest points are well computed."""


        r1 = 0.7
        r2 = 0.5
        
        rmodel = pin.Model()
        cmodel = pin.GeometryModel()
        geometries = [
            hppfcl.Sphere(r1),
            hppfcl.Sphere(r2),
        ]
        
        # With pinocchio3, a new way of constructing a geometry object is available and the old one will be deprecated.
        for i, geom in enumerate(geometries):
            placement = pin.SE3(np.eye(3), np.array([i, 0, 0]))
            try:
                geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, placement, geom)
            except:
                geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, geom, placement)

            cmodel.addGeometryObject(geom_obj)

        rdata = rmodel.createData()
        cdata = cmodel.createData()

        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        # Creating the shapes for the collision detection.
        shape1_id = cmodel.getGeometryId("obj0")

        # Coloring the sphere
        shape1 = cmodel.geometryObjects[shape1_id]
        self.assertIsInstance(shape1.geometry, hppfcl.Sphere)

        # Getting its pose in the world reference
        shape1_placement = cdata.oMg[shape1_id]

        # Doing the same for the second shape.
        shape2_id = cmodel.getGeometryId("obj1")

        # Coloring the sphere
        shape2 = cmodel.geometryObjects[shape2_id]
        self.assertIsInstance(shape2.geometry, hppfcl.Sphere)

        # Getting its pose in the world reference
        shape2_placement = cdata.oMg[shape2_id]

        req = hppfcl.DistanceRequest()
        res = hppfcl.DistanceResult()

        # Testing the distance calculus
        distance_hpp = hppfcl.distance(
            shape1.geometry,
            hppfcl.Transform3f(shape1_placement.rotation, shape1_placement.translation),
            shape2.geometry,
            hppfcl.Transform3f(shape2_placement.rotation, shape2_placement.translation),
            req,
            res,
        )

        distance_ana = np.linalg.norm(shape1_placement.translation - shape2_placement.translation) - (r1 + r2)
        self.assertAlmostEqual(distance_ana, distance_hpp)
        
        # Testing the computation of closest points
        cp1 = res.getNearestPoint1()
        cp2 = res.getNearestPoint2()
        
        distance_cp = -1 * np.linalg.norm(cp1 - cp2) 
        
        # - distance because interpenetration
        self.assertAlmostEqual(distance_cp, distance_ana)
        
        

if __name__ == "__main__":
    unittest.main()
