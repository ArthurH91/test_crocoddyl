# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import pinocchio as pin
import example_robot_data as robex
import hppfcl

# This class is for unwrapping an URDF and converting it to a model. It is also possible to add objects in the model,
# such as a ball at a specific position.


class RobotWrapper:
    def __init__(
        self,
        scale=1.0,
        name_robot="franka",
        urdf_model_path=None,
        mesh_dir=None,
    ):
        """Initialize the wrapper with a scaling number of the target and the name of the robot wanted to get unwrapped.

        Parameters
        ----------
        _scale : float, optional
            Scale of the target, by default 1.0
        name_robot : str, optional
            Name of the robot wanted to get unwrapped, by default "franka"
        """

        self._scale = scale
        self._name_robot = name_robot
        self._urdf_model_path = urdf_model_path
        self._mesh_dir = mesh_dir
        self._color = np.array([249, 136, 126, 255]) / 255

    def __call__(self):
        """Create a robot with a new frame at the end effector position and place a hppfcl: ShapeBase cylinder at this position.

        Parameters
        ----------
        target : bool, optional
            Boolean describing whether the user wants a target or not, by default False

        Returns
        -------
        _robot
            Robot description of the said robot
        _rmodel
            Model of the robot
        _gmodel
            Geometrical model of the robot


        """
        (
            self._rmodel,
            self._collision_model,
            self._visual_model,
        ) = pin.buildModelsFromUrdf(
            self._urdf_model_path, self._mesh_dir, pin.JointModelFreeFlyer()
        )

        q0 = pin.neutral(self._rmodel)

        jointsToLock = [
            "root_joint:",
            "panda2_finger_joint1:",
            "panda2_finger_joint2:",
            "universe",
        ]

        jointsToLockIDs = [1,9,10]

        geom_models = [self._visual_model, self._collision_model]
        self._model_reduced, geometric_models_reduced = pin.buildReducedModel(
            self._rmodel,
            list_of_geom_models=geom_models,
            list_of_joints_to_lock=jointsToLockIDs,
            reference_configuration=q0,
        )

        self._visual_model_reduced, self._collision_model_reduced = (
            geometric_models_reduced[0],
            geometric_models_reduced[1],
        )

        return (
            self._model_reduced,
            self._collision_model_reduced,
            self._visual_model_reduced,
        )

        


if __name__ == "__main__":
    from os.path import dirname, join, abspath
    from wrapper_meshcat import MeshcatWrapper
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)

    # Creating the robot
    robot_wrapper = RobotWrapper(
        name_robot="franka",
        urdf_model_path=urdf_model_path,
        mesh_dir=mesh_dir,
    )
    rmodel, cmodel, vmodel = robot_wrapper()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
        # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(robot_model=rmodel, robot_visual_model=vmodel, robot_collision_model=cmodel)
    vis[0].display(pin.neutral(rmodel))