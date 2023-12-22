from os.path import dirname, join, abspath
import numpy as np
import hppfcl

import pinocchio as pin
try:
    import pydiffcol
except:
    print("pydiffcol not imported")


### HELPERS

YELLOW_FULL = np.array([1, 1, 0, 1.0])
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

WITH_DISPLAY = False


def select_strategy(strat: str, verbose: bool = False):
    """Strategy selection for diffcol. The options are chosen empirically and mostly copied from Louis' examples.

    Args:
        strat (str): strategy of the computation of the derivatives.
        verbose (bool, optional): Defaults to False.

    Raises:
        NotImplementedError: Wrong choice of derivatives

    Returns:
        req: pydiffcol.DistanceRequest.
        req_diff: pydiffcol.DerivativeRequest.
    """
    req = hppfcl.DistanceRequest()
    req.gjk_initial_guess = hppfcl.GJKInitialGuess.CachedGuess
    req.gjk_convergence_criterion = hppfcl.GJKConvergenceCriterion.DualityGap
    req.gjk_convergence_criterion_type = hppfcl.GJKConvergenceCriterionType.Absolute
    req.gjk_tolerance = 1e-8
    req.epa_tolerance = 1e-8
    req.epa_max_face_num = 1000
    req.epa_max_vertex_num = 1000
    req.epa_max_iterations = 1000
    req_diff = pydiffcol.DerivativeRequest()
    req_diff.warm_start = np.array([1.0, 0.0, 0.0])
    req_diff.support_hint = np.array([0, 0], dtype=np.int32)
    req_diff.use_analytic_hessians = True

    if strat == "finite_differences":
        req_diff.derivative_type = pydiffcol.DerivativeType.FiniteDifferences
    elif strat == "zero_order_gaussian":
        req_diff.derivative_type = pydiffcol.DerivativeType.ZeroOrderGaussian
    elif strat == "first_order_gaussian":
        req_diff.derivative_type = pydiffcol.DerivativeType.FirstOrderGaussian
    elif strat == "first_order_gumbel":
        req_diff.derivative_type = pydiffcol.DerivativeType.FirstOrderGumbel
    else:
        raise NotImplementedError

    if verbose:
        print("Strategy: ", req_diff.derivative_type)
        print("Noise: ", req_diff.noise)
        print("Num samples: ", req_diff.num_samples)

    return req, req_diff


def wrapper_robot():
    """Load the robot from the models folder.

    Returns:
        rmodel, vmodel, cmodel: Robot model, visual model & collision model of the robot.
    """
    ### LOADING THE ROBOT
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)
    srdf_model_path = model_path + "/panda/demo.srdf"

    rmodel, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    )

    q0 = pin.neutral(rmodel)
    jointsToLockIDs = [1, 9, 10]

    geom_models = [visual_model, collision_model]
    model_reduced, geometric_models_reduced = pin.buildReducedModel(
        rmodel,
        list_of_geom_models=geom_models,
        list_of_joints_to_lock=jointsToLockIDs,
        reference_configuration=q0,
    )

    visual_model_reduced, collision_model_reduced = (
        geometric_models_reduced[0],
        geometric_models_reduced[1],
    )

    return model_reduced, visual_model_reduced, collision_model_reduced


######################################## DISTANCE & ITS DERIVATIVES COMPUTATION #######################################


def dist(q):
    """Computes the distance with hppfcl. Updates the hppfcl.distanceResult as well with hppfcl.distanceResult.getNearestPoint1() for instance.

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
        shape1.geometry,
        hppfcl.Transform3f(shape1_placement.rotation, shape1_placement.translation),
        shape2.geometry,
        hppfcl.Transform3f(shape2_placement.rotation, shape2_placement.translation),
        req,
        res,
        )
    return distance


def dist_numdiff(q):
    """Finite derivative of the dist function at q.

    Args:
        q (np.ndarray): Configuration of the robot

    Returns:
        distance derivative: distance derivative between shape 1 & shape 2
    """
    j_diff = np.zeros(nq)
    fx = dist(q)
    for i in range(nq):
        e = np.zeros(nq)
        e[i] = 1e-12
        j_diff[i] = (dist(q + e) - dist(q)) / e[i]
    return j_diff


def ddist(q):
    """Diffcol derivative of the dist function with regards to the transformation of the closest point belonging to shape 1.
        The LOCAL frame is used instead of LOCAL_WORLD_ALIGNED. The choice was empyrical. Chosing LOCAL gives the same result as the other derivatives methods.
    Args:
        q (np.ndarray): Configuration of the robot

    Returns:
        distance derivative: distance derivative between shape 1 & shape 2
    """

    # Computing the distance
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    pin.computeJointJacobians(rmodel, rdata, q)

    pydiffcol.distance_derivatives(
        shape1.geometry,
        shape1_placement,
        shape2.geometry,
        shape2_placement,
        req,
        res,
        req_diff,
        res_diff,
    )

    jacobian = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        pin.LOCAL,
    )
    return np.dot(res_diff.ddist_dM1, jacobian)


def derivative_distance_sphere_sphere_florent():
    """Distance derivatives found with the demonstration of florent : https://homepages.laas.fr/florent/publi/05icra1.pdf
     The derivative is computed page 3 and the result is :
     $\frac{\partial d_{ij}}{\partial q} (q) = \frac{( R(q) - O(q))^{T} }{ || O(q) - R(q) ||} \frac{\partial R_{\in body}}{\partial q}$

    Returns:
        distance derivative: distance derivative between shape 1 & shape 2
    """
    pin.forwardKinematics(rmodel, rdata, q)
    pin.computeJointJacobians(rmodel, rdata, q)
    pin.updateGeometryPlacements(
        rmodel,
        rdata,
        cmodel,
        cdata,
        q,
    )
    jacobian = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )

    # Computing the distance
    distance = hppfcl.distance(
        shape1.geometry,
        hppfcl.Transform3f(shape1_placement.rotation, shape1_placement.translation),
        shape2.geometry,
        hppfcl.Transform3f(shape2_placement.rotation, shape2_placement.translation),
        req,
        res,
        )

    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()

    deriv = (cp1 - cp2).T / np.linalg.norm(cp1 - cp2) @ jacobian[:3]

    return deriv


def derivative_distance_sphere_sphere_analytics():
    """Distance derivatives found with the analytical demonstration of the derivative of distance with regards to the translation of the closest point of the shape 1.
    Let S1 & S2 spheres of radius r1 & r2. As spheres are invariable by rotations, let's only work with translations t1 & t2 here. t1 = (x1,x2,x3), where xi $\in$ R3 & t2 = (y1,y2,y3).
    The distance can be written as : $     d (S_1 + t_1, S_2 + t_2) = || t_2 - t_1 || - (r_1 + r_2) $.
    Hence, the derivative : $     \frac{\partial d}{\partial t_1} (S_1 + t_1, S_2 + t_2) = \frac{\partial}{\partial t_1} || t_2 - t_1 || $, 

    The distance can also be written as : $   d (S_1 + t_1, S_2 + t_2) = \sqrt{\sum ^3 _{i = 1} (y_i - x_i)^{2}} $.
    Now, it's only a simple vector derivatives.
    Hence :   $\frac{\partial d}{\partial t_1} (S_1 + t_1, S_2 + t_2) = \sum ^{3}_{i = 1} \frac{(x_i - y_i) e_i }{d(S_1 + t_1, S_2 + t_2)} $ where $ (e_1,e_2,e_3)$ base of $R^3$.
        
    Returns:
        distance derivative: distance derivative between shape 1 & shape 2
    """
    pin.forwardKinematics(rmodel, rdata, q)
    pin.computeJointJacobians(rmodel, rdata, q)
    pin.updateGeometryPlacements(
        rmodel,
        rdata,
        cmodel,
        cdata,
        q,
    )
    jacobian = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )

    # Computing the distance
    distance = hppfcl.distance(
        shape1.geometry,
        hppfcl.Transform3f(shape1_placement.rotation, shape1_placement.translation),
        shape2.geometry,
        hppfcl.Transform3f(shape2_placement.rotation, shape2_placement.translation),
        req,
        res,
        )
    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()

    CP1_SE3 = pin.SE3.Identity()
    CP1_SE3.translation = cp1

    CP2_SE3 = pin.SE3.Identity()
    CP2_SE3.translation = cp2

    deriv = (cp1 - cp2).T / distance @ jacobian[:3]
    return deriv


def dist_numdiff(q):
    """Finite derivative of the dist function with regards to the configuration vector of the robot.

    Args:
        q (np.ndarray): Configuration of the robot

    Returns:
        distance derivative: distance derivative between shape 1 & shape 2
    """
    j_diff = np.zeros(nq)
    fx = dist(q)
    for i in range(nq):
        e = np.zeros(nq)
        e[i] = 1e-5
        j_diff[i] = (dist(q + e) - dist(q)) / e[i]
    return j_diff


if __name__ == "__main__":
    # Loading the robot
    rmodel, vmodel, cmodel = wrapper_robot()
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Initial configuration
    q = np.array([1, 1, 1, 1, 1, 1, 1])
    # q = pin.randomConfiguration(rmodel)
    # Number of joints
    nq = rmodel.nq

    # Updating the models
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    # Creating the shapes for the collision detection.
    shape1_id = cmodel.getGeometryId("panda2_link5_sc_2")

    # Coloring the sphere
    shape1 = cmodel.geometryObjects[shape1_id]
    shape1.meshColor = BLUE_FULL

    # Getting its pose in the world reference
    shape1_placement = cdata.oMg[shape1_id]

    # Doing the same for the second shape.
    shape2_id = cmodel.getGeometryId("panda2_link2_sc_2")

    # Coloring the sphere
    shape2 = cmodel.geometryObjects[shape2_id]
    shape2.meshColor = YELLOW_FULL

    # Getting its pose in the world reference
    shape2_placement = cdata.oMg[shape2_id]

    # Distance & Derivative results from diffcol
    try:
        req, req_diff = select_strategy("finite_differences")
        res = pydiffcol.DistanceResult()
        res_diff = pydiffcol.DerivativeResult()
    except:
        req = hppfcl.DistanceRequest()
        res = hppfcl.DistanceResult()

    # Computing the distance between shape 1 & shape 2 at q0 & comparing with the distance anatically computed
    print(f"dist(q) : {dist(q):.6f}")
    dist2 = np.linalg.norm(shape1_placement.translation - shape2_placement.translation)
    print(
        f"np.linalg.norm(dist2) : {dist2 - shape1.geometry.radius - shape2.geometry.radius:.6f}"
    )

    try:
        deriv_diffcol = ddist(q)
    except:
        print("As pydiffcol is not imported, the derivatives from diffcol won't be computed.")
        
    deriv_ana = derivative_distance_sphere_sphere_analytics()
    deriv_florent = derivative_distance_sphere_sphere_florent()
    deriv_numdiff = dist_numdiff(q)
    np.set_printoptions(precision=3)
    
    try:
        print(f"deriv diffcol : {deriv_diffcol}")
    except:
        pass
    print(f"deriv ana: {deriv_ana}")
    print(f"deriv florent: {deriv_florent}")
    print(f"numdif : {deriv_numdiff}")

    ######### TESTING
    def test():
        # Making sure the shapes exist
        assert shape1_id <= len(cmodel.geometryObjects) - 1
        assert shape2_id <= len(cmodel.geometryObjects) - 1

        # Making sure the shapes are spheres
        assert isinstance(shape1.geometry, hppfcl.Sphere)
        assert isinstance(shape2.geometry, hppfcl.Sphere)

    test()

    ######### VIZ

    if WITH_DISPLAY:
        from wrapper_meshcat import MeshcatWrapper

        # Generating the meshcat visualizer
        MeshcatVis = MeshcatWrapper()
        vis, meshcatVis = MeshcatVis.visualize(
            robot_model=rmodel,
            robot_collision_model=cmodel,
            robot_visual_model=cmodel,
        )
        # Displaying the initial
        vis.display(q)
