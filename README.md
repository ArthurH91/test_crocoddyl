# test_crocoddyl

Scripts to try crocoddyl and mim-solver and their python bindings to do collision detection. 

# Dependencies 

For OCP scripts : 

https://github.com/machines-in-motion/mim_solvers/

https://github.com/loco-3d/crocoddyl

https://github.com/meshcat-dev/meshcat

https://github.com/stack-of-tasks/pinocchio

# Usage 

Before trying the scripts, test your hppfcl installation. To do this and make sure the hppfcl librairy works well in your computer, run : 
``` python src/test_crocoddyl/tests/unit_tests_collisions.py```. This script is for making sure that hppfcl computes well the closest points for the primitive used in this repo.

Then, to try the examples, create a meshcat server using a terminal and the following command : ```meshcat-server```. In another terminal, you can launch for instance ```python src/test_crocoddyl/panda_robot/demo_panda_reaching_obs_single_point.py``` to run the demo.

As the code is still in developpement, the code is constantly moving and sometimes, examples do not work. Hence, do not hesitate to contact me at ahaffemaye@laas.fr. 

# Credits

This repo is based on https://github.com/machines-in-motion/minimal_examples_crocoddyl/tree/master from Sebastien Kleff. 

