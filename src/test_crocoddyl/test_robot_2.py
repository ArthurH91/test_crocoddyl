from os.path import dirname, join, abspath
import numpy as np
import matplotlib.pyplot as plt
import hppfcl

import pinocchio as pin
import pydiffcol

from wrapper_meshcat import MeshcatWrapper

### HELPERS 

YELLOW_FULL = np.array([1, 1, 0, 1.])
BLUE_FULL = np.array([144, 169, 183, 255]) / 255


def select_strategy(strat: str, verbose: bool = False):
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
    req_diff.warm_start = np.array([1., 0., 0.])
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

def dist(q):
    # Computing the distance
    pin.framesForwardKinematics(model_reduced, rdata, q)
    pin.updateGeometryPlacements(
        model_reduced, rdata, collision_model_reduced, cdata, q 
    )

    shape1_placement = cdata.oMg[shape1_id]
    shape2_placement = cdata.oMg[shape2_id]

    distance = pydiffcol.distance(
        shape1_geom,
        shape1_placement,
        shape2_geom,
        shape2_placement,
        req,
        res,
    )
    return distance


def dist_numdiff(q):
    j_diff = np.zeros(nq)
    fx = dist(q)
    for i in range(nq):
        e = np.zeros(nq)
        e[i] = 1e-6
        j_diff[i] = (dist(q + e) - dist(q - e)) / e[i] / 2
    return j_diff

def ddist(q):
    # Computing the distance
    pin.framesForwardKinematics(model_reduced, rdata, q)
    pin.updateGeometryPlacements(
        model_reduced, rdata, collision_model_reduced, cdata, q 
    )

    shape1_placement = cdata.oMg[shape1_id]
    shape2_placement = cdata.oMg[shape2_id]

    pin.computeJointJacobians(model_reduced, rdata, q)

    pydiffcol.distance_derivatives(
        shape1_geom,
        shape1_placement,
        shape2_geom,
        shape2_placement,
        req,
        res,
        req_diff,
        res_diff,
    )

    jacobian = pin.computeFrameJacobian(
        model_reduced,
        rdata,
        q,
        shape1.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )   
    return(np.dot(res_diff.ddist_dM1,jacobian))

if __name__ == "__main__":
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

    # Modifying the collision model to add the capsules
    rdata = model_reduced.createData()
    cdata = collision_model_reduced.createData()
    q0 = pin.neutral(model_reduced)
    q0 = pin.randomConfiguration(model_reduced)

    nq = model_reduced.nq
    # Updating the models
    pin.framesForwardKinematics(model_reduced, rdata, q0)
    pin.updateGeometryPlacements(model_reduced, rdata, collision_model_reduced, cdata, q0)


    shape1_id = collision_model_reduced.getGeometryId("panda2_link7_sc_2")
    shape1 = collision_model_reduced.geometryObjects[shape1_id]
    # Getting the geometry of the shape 1
    shape1_geom = shape1.geometry
    # Getting its pose in the world reference
    shape1_placement = cdata.oMg[shape1_id]
    shape1.meshColor = BLUE_FULL
    # Doing the same for the second shape.
    shape2_id = collision_model_reduced.getGeometryId("panda2_link3_sc_1")
    shape2 = collision_model_reduced.geometryObjects[shape2_id]
    shape2_geom = shape2.geometry
    shape2_placement = cdata.oMg[shape2_id]
    shape2.meshColor = YELLOW_FULL


    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis, meshcatVis = MeshcatVis.visualize(
        robot_model=model_reduced,
        robot_collision_model=collision_model_reduced,
        robot_visual_model=collision_model_reduced,
    )

    vis.display(q0)

    req, req_diff = select_strategy("finite_differences")
    res = pydiffcol.DistanceResult()
    res_diff = pydiffcol.DerivativeResult()


    print(f"dist(q) : {round(dist(q0),6)}")
    dist2 = np.linalg.norm(shape1_placement.translation - shape2_placement.translation)
    print(
        f"np.linalg.norm(dist2) : {round(np.linalg.norm(dist2) - shape1_geom.radius - shape2_geom.radius,6)}"
    )


    l = []
    ddist_list = []
    ddist_numdiff_list = []
    alpha = np.linspace(-np.pi,np.pi,10000)
    for k in alpha:
        q = np.array([0,0,0,k,0,0,0])
        d = dist(q)  
        l.append(d)
        
        ddist_diffcol = ddist(q)
        ddist_numdiff = dist_numdiff(q)
        ddist_list.append(ddist_diffcol)
        ddist_numdiff_list.append(ddist_numdiff)

    plots = [331, 332, 333, 334, 335, 336, 337]
    for k in range(nq):
        plt.subplot(plots[k])
        plt.plot(alpha, np.array(ddist_list)[:,k],"--" , label = "diffcol")
        plt.plot(alpha, np.array(ddist_numdiff_list)[:, k], label = "numdiff")
        plt.title("joint" + str(k))
        plt.legend()
    plt.show()