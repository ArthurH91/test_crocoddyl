import os
from os.path import dirname, join, abspath
import argparse
import json, codecs
import time
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

import hppfcl

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper

from utils import get_transform, linear_gradient, check_limits, get_transform_from_list

###* PARSERS

###* HYPERPARAMS
T = 10
nq = 7

name = "single_wall_without_autocollision_but_obstacle_constraints_non_neutral_start_only_translation_reaching"
name = "single_wall_with_some_constraints_and_obstacle_non_neutral_start_only_translation_reaching" #! NO CONVERGENCE
name = "single_wall_with_all_constraints_non_neutral_start_only_translation_reaching"
name = "single_wall_without_any_constraints_non_neutral_start_only_translation_reaching"
name = "single_wall_without_constraints_obstacle_non_neutral_start_only_translation_reaching" #! SAME RESULT (TOUCHES OBSTACLE)
name = "single_wall_with_all_constraints"

# name = "big_wall_in_front_robot_constraints_neutral_start_only_translation_reaching" 
# name = "big_wall_with_some_constraints_and_obstacle_neutral_start_only_translation_reaching" #! SAME RESULT (GOES THROUGH THE OBSTACLE)



### LOADING THE ROBOT
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)
srdf_model_path = model_path + "/panda/demo.srdf"

# Creating the robot
robot_wrapper = RobotWrapper(
    urdf_model_path=urdf_model_path, mesh_dir=mesh_dir, srdf_model_path=srdf_model_path
)
rmodel, cmodel, vmodel = robot_wrapper()
rdata = rmodel.createData()


# Openning the files
path = os.getcwd()

results_json = codecs.open(
    path + "/results/" + name + ".json", "r", encoding="utf-8"
).read()

results = json.loads(results_json)
OBSTACLE_POSE = results["obstacle_pose"]
TARGET_POSE = results["target_pose"]
OBSTACLE_DIM = results["obstacle_dim"]

if results["type"] == "single_wall":
    # OBSTACLE_SHAPE = hppfcl.Box(OBSTACLE_DIM)
    OBSTACLE_TYPE = "box"

print(OBSTACLE_POSE)
print(get_transform_from_list(OBSTACLE_POSE))
# # SETTING UP THE VISUALIZER
MeshcatVis = MeshcatWrapper()
vis, meshcatvis = MeshcatVis.visualize(
    get_transform_from_list(TARGET_POSE),
    get_transform_from_list(OBSTACLE_POSE),
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
    obstacle_type=OBSTACLE_TYPE,
    OBSTACLE_DIM=OBSTACLE_DIM,
)

# VISUALIZING THE RESULTS
while True:
    for q in results["Q"]:
        vis.display(np.array(q))
        time.sleep(1e-2)
    input("Press a key for replay")