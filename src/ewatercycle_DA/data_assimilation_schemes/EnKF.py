from typing import Any, Optional
from pydantic import BaseModel
import numpy as np

from ewatercycle_DA.utils import add_normal_noise

class EnsembleKalmanFilter(BaseModel):
    """Implementation of an Ensemble Kalman filter scheme to be applied to the :py:class:`Ensemble`.

    note:
        The :py:class:`EnsembleKalmanFilter` is controlled by the :py:class:`Ensemble` and thus has no time reference itself.
        No DA method should need to know where in time it is (for now).
        Currently assumed 1D grid.

    Args:
        hyperparameters (dict): Combination of many different parameters:
                                like_sigma_weights (float): scale/sigma of logpdf when generating particle weights

                                like_sigma_state_vector (float): scale/sigma of noise added to each value in state vector

    Attributes:
        obs (float): observation value of the current model timestep, set in due course thus optional

        state_vectors (np.ndarray): state vector per ensemble member [N x len(z)]

        predictions (np.ndarray): contains prior modeled values per ensemble member [N x 1]

        new_state_vectors (np.ndarray): updated state vector per ensemble member [N x len(z)]

        All are :obj:`None` by default
    """

    hyperparameters: dict = dict(like_sigma_state_vector=0.0005, observed_state_index=1)
    N: int
    obs: Optional[Any | None] = None # np.ndarray
    state_vectors: Optional[Any | None] = None
    predictions: Optional[Any | None] = None
    new_state_vectors: Optional[Any | None] = None
    logger: list = [] # easier than using built in logger ?


    def update(self):
        """Takes current state vectors of ensemble and returns updated state vectors ensemble

        TODO: refactor to be more readable
        """

        # TODO: is obs are not float but array should be mXN, currently m = 1: E should be mxN, D should be m x N
        measurement_d = np.matrix(self.obs).T

        measurement_pertubation_matrix_E = np.array([[add_normal_noise(self.hyperparameters['like_sigma_state_vector'])] * measurement_d.shape[0] for _ in range(self.N)]).T

        peturbed_measurements_D = measurement_d * np.matrix(np.ones(self.N).T) + np.sqrt(
                                                                        self.N - 1) * measurement_pertubation_matrix_E

        predicted_measurements_Ypsilon = np.matrix(self.predictions).T
        prior_state_vector_Z = self.state_vectors.T

        PI = np.matrix((np.identity(self.N) - ((np.ones(self.N) @ np.ones(self.N).T) / self.N)) / (
            np.sqrt(self.N - 1)))
        A_cross_A = np.matrix(
            (np.identity(self.N) - ((np.ones(self.N) @ np.ones(self.N).T) / self.N)))


        E = np.matrix(peturbed_measurements_D) * PI

        Y = np.matrix(predicted_measurements_Ypsilon) * PI
        if prior_state_vector_Z.shape[0] < self.N - 1:
            Y = Y * A_cross_A
        S = Y
        self.logger.append(f'{peturbed_measurements_D.shape}, {predicted_measurements_Ypsilon.shape}')

        D_tilde = peturbed_measurements_D - predicted_measurements_Ypsilon

        self.logger.append(f'PI{PI.shape},E{E.shape}, Y{Y.shape}, D_tilde{D_tilde.shape}')
        W = (S.T * np.linalg.inv(S * S.T + E * E.T)) * D_tilde
        T = np.identity(self.N) + (W / np.sqrt(self.N - 1))

        # only update observed state
        new_state_vectors = prior_state_vector_Z.copy()
        new_state_vectors[self.hyperparameters['observed_state_index'], :] = (prior_state_vector_Z[self.hyperparameters['observed_state_index'], :] * T)

        self.new_state_vectors = np.array(new_state_vectors.T) # back to N x len(z) to be set correctly



