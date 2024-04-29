from typing import Any
from pydantic import BaseModel

import random
import numpy as np
import scipy

from ewatercycle_DA.utils import add_normal_noise
class ParticleFilter(BaseModel):
    """Implementation of a particle filter scheme to be applied to the :py:class:`Ensemble`.

    note:
        The :py:class:`ParticleFilter` is controlled by the :py:class:`Ensemble` and thus has no time reference itself.
        No DA method should need to know where in time it is (for now).
        Currently assumed 1D grid.

    Args:
        N (int): Size of ensemble, passed down from DA.Ensemble().

    Attributes:
        hyperparameters (dict): Combination of many different parameters:
                                like_sigma_weights (float): scale/sigma of logpdf when generating particle weights

                                like_sigma_state_vector (float): scale/sigma of noise added to each value in state vector

                                f_n_particles (float): factor between 0 and 1 which sets the threshold for when to resample.
                                                        if N_eff < f_n_particles * N then resample

        obs (float): observation value of the current model timestep, set in due course thus optional

        state_vectors (np.ndarray): state vector per ensemble member [N x len(z)]

        predictions (np.ndarray): contains prior modeled values per ensemble member [N x 1]

        new_state_vectors (np.ndarray): updated state vector per ensemble member [N x len(z)]

        weights (np.ndarray): contains weights per ensemble member per prior modeled values [N x 1]

        resample_indices (np.ndarray): contains indices of particles that are resampled [N x 1]


    All are :obj:`None` by default


    """
    # args
    N: int

    # required attributes
    hyperparameters: dict = dict(like_sigma_weights=0.05,
                                 like_sigma_state_vector=0.0005,
                                  f_n_particles=0.8)
    obs: float | Any | None = None # TODO: refactor to np.ndarray
    state_vectors: Any | None = None
    predictions: Any | None = None
    new_state_vectors: Any | None = None

    # extra attributes
    weights: Any | None = None
    resample_indices: Any | None = None
    resample: bool = False
    N_eff: float | None = None


    def update(self):
        """Takes current state vectors of ensemble and returns updated state vectors ensemble
        """
        self.generate_weights()
        # if np.isnan(self.weights).all():
        #     like_sigma_original = self.hyperparameters['like_sigma_weights']
        #     max_iter, i = 50, 1
        #     while np.isnan(self.weights.mean()) and i < max_iter:
        #         self.hyperparameters['like_sigma_weights'] *= 2
        #         self.generate_weights()
        #         i+=1
        #     self.hyperparameters['like_sigma_weights'] = like_sigma_original
        # TODO: Refactor to be more modular i.e. remove if/else

        self.N_eff = 1 / (self.weights**2).sum()
        resample_threshold = self.hyperparameters['f_n_particles'] * self.N

        if self.N_eff < resample_threshold:
            self.resample = True
        else:
            self.resample = False

        if self.resample:
            # 1d for now: weights is N x 1: in the case of HBV
            if self.weights[0].size == 1:
                self.resample_indices = random.choices(population=np.arange(self.N), weights=self.weights, k=self.N)

                new_state_vectors = self.state_vectors.copy()[self.resample_indices]
                new_state_vectors_transpose = new_state_vectors.T # change to len(z) x N so in future you can vary sigma
            # 2d weights is N x len(z)
            else:
                # handel each row separately:
                self.resample_indices = []
                for i in range(len(self.weights[0])):
                     self.resample_indices.append(random.choices(population=np.arange(self.N), weights=self.weights[:, i], k=self.N))
                self.resample_indices = np.vstack(self.resample_indices)

                new_state_vectors_transpose = self.state_vectors.copy().T
                for index, indices in enumerate(self.resample_indices):
                    new_state_vectors_transpose[index] = new_state_vectors_transpose[index, indices]


            like_sigma = self.hyperparameters['like_sigma_state_vector']
            # for now just constant perturbation, can vary this hyperparameter
            if type(like_sigma) is float:
                for index, row in enumerate(new_state_vectors_transpose):
                    row_with_noise = np.array([s + add_normal_noise(like_sigma) for s in row])
                    new_state_vectors_transpose[index] = row_with_noise

            elif type(like_sigma) is list and len(like_sigma) == len(new_state_vectors_transpose):
                for index, row in enumerate(new_state_vectors_transpose):
                    row_with_noise = np.array([s + add_normal_noise(like_sigma[index]) for s in row])
                    new_state_vectors_transpose[index] = row_with_noise
            else:
                raise RuntimeWarning(f"{like_sigma} should be float or list of length {len(new_state_vectors_transpose)}")

            self.new_state_vectors = new_state_vectors_transpose.T  # back to N x len(z) to be set correctly

        # is not resampling, also don't update state vector
        else:
            self.new_state_vectors = self.state_vectors


    def generate_weights(self):
        """Takes the ensemble and observations and returns the posterior"""

        like_sigma = self.hyperparameters['like_sigma_weights']
        difference = (self.obs - self.predictions)
        unnormalised_log_weights = scipy.stats.norm.logpdf(difference, loc=0, scale=like_sigma)
        normalised_weights = np.exp(unnormalised_log_weights - scipy.special.logsumexp(unnormalised_log_weights))

        self.weights = normalised_weights
