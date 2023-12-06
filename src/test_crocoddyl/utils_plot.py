import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import time
import pydiffcol
import hppfcl 

from utils import select_strategy
from diff_finie_diffcol import red, green, yellow

import meshcat

import meshcat.geometry as g
import meshcat.transformations as tf

from os.path import dirname, join, abspath

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper

from utils import get_transform

def plot_costs(rd):
    """Plot all the costs of the problem.

    Args:
        rd (runningData): Running Data of crocoddyl.
    """
    ### Construction of the dictionnary
    costs_dict = {}
    # Filling the dictionnary with empty entries of the name of the costs
    for name in rd[0].differential.costs.costs.todict():
        costs_dict[name] = []
    # Filling the lists of the dictionnary with the values of the said costs.
    for name in costs_dict:
        for data in rd:
            costs_dict[name].append(data.differential.costs.costs[name].cost)

    ### Plotting
    for name_cost in costs_dict:
        # Plotting the collisions costs with "-o"
        if "col" in name_cost:
            plt.plot(costs_dict[name_cost], "-o", label=name_cost, markersize=3)
        else:
            plt.plot(costs_dict[name_cost], "o", label=name_cost, markersize=3)
    plt.xlabel("Nodes")
    plt.ylabel("Cost (log)")
    plt.legend()
    plt.yscale("log")
    plt.show()


def display_with_col(Q: list, vis, meshcatvis, rmodel, rdata, cmodel, cdata ):
    obstacle_id = cmodel.getGeometryId("obstacle")
    for i,q in enumerate(Q):
        for k in range(len(cmodel.collisionPairs)):
            if obstacle_id == cmodel.collisionPairs[k].first or obstacle_id==cmodel.collisionPairs[k].second:
                col, w1, w2 = check_collision(rmodel, rdata, cmodel, cdata, q, k, i)
                if "ok" in col:
                    color = green
                if "almost" in col:
                    color = yellow
                if "collision" in col:
                    color = red
                r1 = 5e-3
                meshcatvis["cp" + str(k) + str(i)].set_object(g.Sphere(r1), color)
                T = pin.SE3.Identity()
                T.translation = w1
                meshcatvis["cp" + str(k) + str(i)].set_transform(get_transform(T))
            else:
                _ = check_collision(rmodel, rdata, cmodel, cdata, q, k, i)
                
        vis.display(q)
        time.sleep(1e-3)


def check_collision(rmodel, rdata, cmodel, cdata, q, pair_id, i ):
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

    # Distance Request & Result from hppfcl / pydiffcol
    req, req_diff = select_strategy("first_order_gaussian")
    res = pydiffcol.DistanceResult()
    res_diff = pydiffcol.DerivativeResult()

    # Getting the ID of the first shape from the collision pair id
    shape1_id = cmodel.collisionPairs[pair_id].first

    # Getting its geometry
    shape1_geom = cmodel.geometryObjects[shape1_id].geometry

    # Getting its pose
    shape1_placement = cdata.oMg[shape1_id]

    # Doing the same for the second shape.
    shape2_id = cmodel.collisionPairs[pair_id].second
    shape2_geom = cmodel.geometryObjects[shape2_id].geometry
    shape2_placement = cdata.oMg[shape2_id]

    # Computing the distance
    d = pydiffcol.distance(
        shape1_geom,
        shape1_placement,
        shape2_geom,
        shape2_placement,
        req,
        res,
    )
    if d > -1e-2 and d<0:
        
        print(f"collision at the {i}-th step. Between : {cmodel.geometryObjects[shape1_id].name} & {cmodel.geometryObjects[shape2_id].name} Value of distance =  {np.round(d,5)}")
        return("collision", res.w1, res.w2)
    if d < -1e3:
        print(f"ACCEPTABLE collision at the {i}-th step. Between : {cmodel.geometryObjects[shape1_id].name} & {cmodel.geometryObjects[shape2_id].name} Value of distance =  {np.round(d,5)}")
        return("collision", res.w1, res.w2)
    elif d <= 0.05 and d >= 0:
        return("almost", res.w1, res.w2)
    else:
        return("ok", res.w1, res.w2)


if __name__ == "__main__":
    ###* LOADING THE ROBOT
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)
    srdf_model_path = model_path + "/panda/demo.srdf"

    ###* Creating the robot
    robot_wrapper = RobotWrapper(
        urdf_model_path=urdf_model_path, mesh_dir=mesh_dir, srdf_model_path=srdf_model_path
    )
    rmodel, cmodel, vmodel = robot_wrapper()
    rdata = rmodel.createData()
    
        # Target pose
    TARGET = np.array([0, 0, 0.85])
    TARGET_POSE = pin.SE3.Identity()
    TARGET_POSE.translation = TARGET
    TARGET_POSE.rotation = pin.utils.rotate("x", np.pi)

    # Creation of the obstacle
    OBSTACLE_DIM = 2e-1
    OBSTACLE = hppfcl.Sphere(OBSTACLE_DIM)
    OBSTACLE_POSE = pin.SE3.Identity()
    OBSTACLE_POSE.translation = np.array([0.32, 0., 1.5])

    # INITIAL_CONFIG = np.array([0, 0, 0, -0.5, 0, 1, 0])
    INITIAL_CONFIG = np.array([0, 0, 0, 0, 0, 0, 0])
    
    

    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
        OBSTACLE,
        OBSTACLE_POSE,
    )
    IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

    ###* ADDING THE COLLISION PAIRS 
    for k in range(16, 23):
        cmodel.addCollisionPair(pin.CollisionPair(k, IG_OBSTACLE))

    cdata = cmodel.createData()

    for k in range(len(cmodel.collisionPairs)):
        print(f"first object : {cmodel.geometryObjects[cmodel.collisionPairs[k].first].name}, second object : {cmodel.geometryObjects[cmodel.collisionPairs[k].second].name}")
        print(f"check collision : {check_collision(rmodel, rdata, cmodel, cdata, INITIAL_CONFIG, k)}")

    # Removing the obstacle of the geometry model because 
    cmodel.removeGeometryObject("obstacle")

    # SETTING UP THE VISUALIZER
    MeshcatVis = MeshcatWrapper()
    vis, meshcatvis = MeshcatVis.visualize(
        TARGET_POSE,
        OBSTACLE_POSE,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
        obstacle_type="sphere",
        OBSTACLE_DIM=OBSTACLE_DIM,
    )
    vis.display(INITIAL_CONFIG)
    
    