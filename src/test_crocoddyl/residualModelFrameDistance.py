from copy import copy
import numpy as np
import pinocchio as pin
import crocoddyl
from crocoddyl.utils import *
import pydiffcol
from utils import select_strategy


class ResidualModelFrameDistance(crocoddyl.ResidualModelAbstract):
    def __init__(
                self,
        state,
        geom_model: pin.Model,
        geom_data,
        OBSTACLE_TRANSLATION: int,
    ):
        
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, True, True, True)
        
        self.nq = self.state.nq 
        self.pinocchio = self.state.pinocchio
        self.geom_model = geom_model
        self.geom_data = geom_data
        self._eps = 5e-3
        self._OBSTACLE_TRANSLATION = OBSTACLE_TRANSLATION

    def calc(self, data, x, u=None):
        
        # Storinr q outside of the state vector
        self._endeff_translation = data.shared.pinocchio.oMf["panda2_leftfinger"].translation
        
        dist = 1/2 * np.linalg.norm(self._endeff_translation - self._OBSTACLE_TRANSLATION)**2
        data.r[:] = dist
        
    def calcDiff(self, data, x, u=None):
        
        
        dee_dq = data.shared.pinocchio.oMf['panda2_leftfinger'].rotation
        data.Rx[:self.nq] = dee_dq.T @ (self._endeff_translation - self._OBSTACLE_TRANSLATION)


if __name__ == "__main__":
    pass
