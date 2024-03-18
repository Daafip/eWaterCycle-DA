"""Class of functions to integrate Data Assimilation into eWaterCycle

Note:
    assumes a 1D grid currently (e.g. in ``get_state_vector``) - not yet tested on distributed models.
"""

import random
import warnings

import scipy
import numpy as np
import xarray as xr
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel
from ewatercycle.base.forcing import DefaultForcing
import ewatercycle
from ewatercycle.models import HBV, Lorenz

LOADED_MODELS: dict[str, Any] = dict(
                                        HBV=ewatercycle.models.HBV,
                                        Lorenz = ewatercycle.models.Lorenz,
                                     )
LOADED_HYDROLOGY_MODELS: dict[str, Any] = dict(
                                        HBV=ewatercycle.models.HBV)
TLAG_MAX = 100 # sets maximum lag possible
class Ensemble(BaseModel):
    """Class for running data assimilation in eWaterCycle

    Args:
        N : Number of ensemble members

        location : Where the model is run, by default local - change to remote later

    Attributes:
        ensemble_method: method used for data assimilation

        ensemble_method_name: name of method used for data assimilation (needed for function specific)

        ensemble_list : list containing ensembleMembers

        observed_variable_name: Name of the observed value: often Q but could be anything

        observations: NetCDF file containing observations

        lst_models_name: list containing a set of all the model names: i.e. to run checks


    Note:
        Run ``setup`` and ``initialize`` before using other functions

    """

    N: int
    location: str = "local"

    ensemble_list: list = []
    ensemble_method: Any | None = None
    ensemble_method_name: str | None = None
    observed_variable_name: str | None = None
    prediction_variable_name: str | None = None
    observations: Any | None = None
    lst_models_name: list = []
    logger: list = [] # logging proved too complex for now so just append to list XD


    def setup(self) -> None:
        """Creates a set of empty Ensemble member instances
        This allows further customisation: i.e. different models in one ensemble
        """
        if len(self.ensemble_list) != 0:
            self.ensemble_list = []
        for ensemble_member in range(self.N):
            self.ensemble_list.append(EnsembleMember())

    def initialize(self, model_name, forcing, setup_kwargs, state_vector_variables='all') -> None:
        """Takes empty Ensemble members and launches the model for given ensemble member

        Args:
            model_name (str | list): just takes the modl string name for now, change to more formal config file later?
                Should you pass a list here, you also need a list of forcing objects & vice versa.

            forcing (:obj:`ewatercycle.base.forcing.DefaultForcing` | :obj:`list`): object or list of objects to give to the model.
                Should you want to vary the forcing, you also need to pass a list of models.

            setup_kwargs (:obj:`dict` | :obj:`list`): kwargs dictionary which can be passed as `model.setup(**setup_kwargs)`.
                UserWarning: Ensure your model saves all kwargs to the config
                Should you want to vary initial parameters, again all should be a list

            state_vector_variables (str | :obj:`list[str]`): if not specified: by default 'all' known parameters,
                can be a subset of all by passing a list containing strings of variable to include in the state vector.
                Should you want to vary initial parameters, again all should be a list

        Note:
            If you want to pass a list for any one variable, **all** others should be lists too of the same length.
        """

        # same for all members (to start with)
        if type(model_name) == str:
            for ensemble_member in self.ensemble_list:
                ensemble_member.model_name = model_name
                ensemble_member.forcing = forcing
                ensemble_member.setup_kwargs = setup_kwargs
                ensemble_member.state_vector_variables = state_vector_variables

        # more flexibility - could change in the future?
        elif type(model_name) == list and len(model_name) == self.N:
            validity_initialize_input(model_name, forcing, setup_kwargs, state_vector_variables)
            for index_m, ensemble_member in enumerate(self.ensemble_list):
                ensemble_member.model_name = model_name[index_m]
                ensemble_member.forcing = forcing[index_m]
                ensemble_member.setup_kwargs = setup_kwargs[index_m]
                ensemble_member.state_vector_variables = state_vector_variables[index_m]
        else:
            raise SyntaxWarning(f"model should either string or list of string of length {self.N}")

        # setup & initialize - same in both cases
        for ensemble_member in self.ensemble_list:
            ensemble_member.verify_model_loaded()
            ensemble_member.setup()
            ensemble_member.initialize()
            self.lst_models_name.append(ensemble_member.model_name)

        self.lst_models_name = list(set(self.lst_models_name))

    def initialize_method(self,
                          ensemble_method_name: str,
                          observation_path: Path,
                          observed_variable_name: str,
                          hyper_parameters: dict,
                          prediction_variable_name: str | None = None,
                          ):
        """Similar to initialize but specifically for the data assimilation method

        Args:
            ensemble_method_name (str): name of the data assimilation method for the ensemble

            observation_path (Path): Path to a NetCDF file containing observations.
                Ensure the time dimension is of type :obj:`numpy.datetime64[ns]` in order to work well with
                 .. py:function:: `Ensemble.update`

            observed_variable_name (str): Name of the observed value: often Q but could be anything

            hyper_parameters (dict): dictionary containing hyperparameters for the method, these will vary per method
                and thus are merely passed on

            prediction_variable_name (Optional[str]):is observed variable name is different from prediction variable name,
                this can be used to fix that. In most cases it is easiest is to change your observation name but not always possible.

        Note:
            Assumed memory is large enough to hold observations in memory/lazy open with xarray
        """
        validate_method(ensemble_method_name)
        self.ensemble_method = LOADED_METHODS[ensemble_method_name](N=self.N)
        self.ensemble_method_name = ensemble_method_name
        self.observed_variable_name = observed_variable_name
        self.set_prediction_variable_name(prediction_variable_name)
        self.observations = self.load_netcdf(observation_path, observed_variable_name)
        for hyper_param in hyper_parameters:
            self.ensemble_method.hyperparameters[hyper_param] = hyper_parameters[hyper_param]
        self.ensemble_method.N = self.N

    def set_prediction_variable_name(self, prediction_variable_name) -> None:
        """In some cases the variable name you predict is different from the observed.
           easiest is to change your observation name but not always possible"""
        if prediction_variable_name is None:
            self.prediction_variable_name = self.observed_variable_name
        else:
            self.prediction_variable_name = prediction_variable_name

    def finalize(self) -> None:
        """Runs finalize step for all members"""
        for ensemble_member in self.ensemble_list:
            ensemble_member.finalize()

    def update(self, assimilate=True) -> None:
        """Updates model for all members.
        Args:
            assimilate (bool): Whether to assimilate in a given timestep. True by default.
        Algorithm flow:
            Gets the state vector, modeled outcome and corresponding observation

            Computes new state vector using supplied method

            Then set the new state vector

        Currently assumed 1D: only one observation per timestep converted to float

         Todo: think about assimilation windows not being every timestep
         """

        # you want the observation before you advance the model, as ensemble_member.update() already advances
        # as day P & E of 0 correspond with Q of day 0. -
        # # but breaks with other implementations?

        for ensemble_member in self.ensemble_list:
            ensemble_member.update()

        if assimilate:
            self.ensemble_method.state_vectors = self.get_state_vector()
            self.logger.append(f'state_vector = {self.ensemble_method.state_vectors.shape}')

            # get observations
            current_time = np.datetime64(self.ensemble_list[0].model.time_as_datetime)
            current_obs = self.observations.sel(time=current_time, method="nearest").values
            self.ensemble_method.obs = current_obs

            # collect predicted model outcome & pass on to ensemble method
            # TODO: these are currently presumed equal, but not per se always so: incase of streamflow -> yes
            # TODO: in case of the lorenz model for example no
            # TODO: as this is more a 1d approach

            self.ensemble_method.predictions = self.get_value(self.prediction_variable_name)

            self.ensemble_method.update()

            self.remove_negative()

            self.set_state_vector(self.ensemble_method.new_state_vectors)

            self.config_specific_actions()


    def remove_negative(self):
        """if only one model is loaded & hydrological: sets negative numbers to positive"""
        if len(self.lst_models_name) == 1 and self.lst_models_name[0] in LOADED_HYDROLOGY_MODELS:
                # set any values below 0 to small
                self.ensemble_method.new_state_vectors[self.ensemble_method.new_state_vectors < 0] = 1e-6
        else:
            warnings.warn("More than 1 model type loaded, no non zero values removes",category=UserWarning)

    def config_specific_actions(self):
        """Function for actions which are specific to a combination of model with method.

            Note:
                Be specific when these are used to only occur when wanted

            *#1: PF & HBV*:
                Particle filters replace the full particle: thus the lag function also needs to be copied.

                If only HBV models are implemented with PF this will be updates

                if HBV and other models are implemented, this will present a RuntimeWarning.

                If other models are implemented with PF, nothing should happen, just a UserWarning so you're aware.


        """

        #1
        if self.ensemble_method_name == "PF":
            # in particle filter the whole particle needs to be copied
            # when dealing with lag this is difficult as we don't want it in the regular state vector

            if "HBV" in self.lst_models_name and len(self.lst_models_name) == 1:
                # first get the memory vectors for all ensemble members
                lag_vector_arr = np.zeros((len(self.ensemble_list),TLAG_MAX))
                for index, ensemble_member in enumerate(self.ensemble_list):
                    t_lag = int(ensemble_member.get_value("Tlag")[0])
                    old_t_lag = np.array([ensemble_member.get_value(f"memory_vector{i}") for i in range(t_lag)]).flatten()
                    lag_vector_arr[index,:t_lag] = old_t_lag
                # resample so has the correct state
                # TODO consider adding noise ?
                new_lag_vector_lst = lag_vector_arr[self.ensemble_method.resample_indices]

                for index, ensembleMember in enumerate(self.ensemble_list):
                    new_t_lag = ensembleMember.get_value(f"Tlag")
                    [ensembleMember.set_value(f"memory_vector{mem_index}", np.array([new_lag_vector_lst[index, mem_index]])) for mem_index in range(int(new_t_lag))]

            elif "HBV" in self.lst_models_name:
                warnings.warn(f"Models implemented:{self.lst_models_name}, could cause issues with particle filters"
                              'HBV needs to update the lag vector but cannot due to other model type(s)',
                              category=RuntimeWarning)
            else:
                warnings.warn(f"Not running `config_specific_actions`",category=UserWarning)

        #2...

    def get_value(self, var_name: str) -> np.ndarray:
        """Gets current value of whole ensemble for given variable ### currently assumes 2d, fix for 1d:"""
        # infer shape of state vector:
        ref_model = self.ensemble_list[0]
        shape_data = ref_model.get_value(ref_model.variable_names[0]).shape[0]
        # shape_var = len(self.variable_names)


        output_array = np.zeros((self.N,shape_data))

        self.logger.append(f'{output_array.shape}')
        for i, ensemble_member in enumerate(self.ensemble_list):
            output_array[i] = ensemble_member.model.get_value(var_name)
        return output_array

    def get_state_vector(self) -> np.ndarray:
        """Gets current value of whole ensemble for specified state vector
            Note:
                Assumes 1d array? although :obj:`np.vstack` does work for 2d arrays
        """
        # collect state vector
        output_lst = []
        for ensemble_member in self.ensemble_list:
            output_lst.append(ensemble_member.get_state_vector())
        return np.vstack(output_lst) # N x len(z)

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        """Sets current value of whole ensemble for given variable
            args:
                src (np.ndarray): size = number of ensemble members x 1 [N x 1]
        """
        for i, ensemble_member in enumerate(self.ensemble_list):
            ensemble_member.model.set_value(var_name, src[i])

    def set_state_vector(self, src: np.ndarray) -> None:
        """Sets current value of whole ensemble for specified state vector

            args:
                src (np.ndarray): size = number of ensemble members x number of states in state vector [N x len(z)]
                    src[0] should return the state vector for the first value
        """
        self.logger.append(f'size_state_vector: {src.shape}')
        for i, ensemble_member in enumerate(self.ensemble_list):
            self.logger.append(src[i])
            ensemble_member.set_state_vector(src[i])

    @staticmethod
    def load_netcdf(observation_path: Path, observed_variable_name: str) -> xr.DataArray:
        """Load the observation data file supplied by user"""
        data = xr.open_dataset(observation_path)
        try:
            assert "time" in data.dims
        except AssertionError:
            raise UserWarning(f"time not present in NetCDF file presented")

        try:
            assert observed_variable_name in data.data_vars
        except AssertionError:
            raise UserWarning(f"{observed_variable_name} not present in NetCDF file presented")

        return data[observed_variable_name]


class EnsembleMember(BaseModel):
    """Class containing ensemble members, meant to be called by the DA.Ensemble class

    Args:
        model_name (str | list[str]): just takes the modl string name for now, change to more formal config file later?
            Should you pass a list here, you also need a list of forcing objects & vice versa.

        forcing (:obj:`ewatercycle.base.forcing.DefaultForcing` | :obj:`list`): object or list of objects to give
            to the model. Should you want to vary the forcing, you also need to pass a list of models.

        setup_kwargs (:obj:`dict` | :obj:`list[dict]`): kwargs dictionary which can be passed as
            `model.setup(**setup_kwargs)`. UserWarning: Ensure your model saves all kwargs to the config
            Should you want to vary initial parameters, again all should be a list

        state_vector_variables (str | :obj:`list[str]`): By default 'all' known parameters,
            can be a subset of all by passing a list containing strings of variable to include in the state vector.
            Changing to a subset allows you to do interesting things with ensembles: mainly limited to particle filters.
            For example giving half the particle filters more variables which vary than others - see what that does.


    Attributes:
        model (:obj:`ewatercycle.base.model`): instance of eWaterCycle model to be used.
            Must be defined in ``loaded_models`` dictionary in this file which is a safeguard against misuse.

        config (:obj:`Path`): path to config file for the model which the EnsembleMember contains.

        state_vector (:obj:`np.ndarray`): numpy array containing last states which were gotten

        variable_names (list[str]): list of string containing the variables in the state vector.

    """

    model_name: str | None = None
    forcing: DefaultForcing | None = None
    setup_kwargs: dict | None = None
    state_vector_variables: str | list = "all"

    model: Any | None = None
    config: Path | None = None
    state_vector: Any | None = None
    variable_names: list[str] | None = None

    def setup(self) -> None:
        """Setups the model provided with forcing and kwargs. Set the config file"""
        self.model = LOADED_MODELS[self.model_name](forcing=self.forcing)
        self.config, _ = self.model.setup(**self.setup_kwargs)

    def initialize(self) -> None:
        """Initializes the model with the config file generated in setup"""
        self.model.initialize(self.config)

        # set correct variable names
        if self.state_vector_variables == "all":
            self.variable_names = list(dict(self.model.parameters).keys()) + list(dict(self.model.states).keys())

        elif type(self.state_vector_variables) == list:
            self.variable_names = self.state_vector_variables

        else:
            raise UserWarning(f"Invalid input state_vector_variables: {self.state_vector_variables}"\
                              +"Must be 'all' or list[str] containing wanted variables.")


    def get_value(self, var_name: str) -> np.ndarray:
        """gets current value of an ensemble member"""
        return self.model.get_value(var_name)

    def get_state_vector(self) -> np.ndarray:
        """Gets current state vector of ensemble member
        Note: assumed a 1D grid currently as ``state_vector`` is 1D array.
        Now check the shape of data and variables.

        """
        # infer shape of state vector:
        shape_data = self.get_value(self.variable_names[0]).shape[0]
        shape_var = len(self.variable_names)

        self.state_vector = np.zeros((shape_var, shape_data))
        for v_index, var_name in enumerate(self.variable_names):
            self.state_vector[v_index] = self.get_value(var_name)
        # changing to fit 2d, breaks 1d... better fix later:
        if shape_data == 1:
            self.state_vector = self.state_vector.T

        return self.state_vector

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        """Sets current value of an ensemble member"""
        self.model.set_value(var_name, src)

    def set_state_vector(self,src: np.ndarray) -> None:
        """Sets current state vector of ensemble member
        Note: assumes a 1D grid currently as ``state_vector`` is 1D array.
        """
        for v_index, var_name in enumerate(self.variable_names):
            self.set_value(var_name, src[v_index])

    def finalize(self) -> None:
        """"Finalizes the model: closing containers etc. if necessary"""
        self.model.finalize()

    def update(self) -> None:
        """Updates the model to the next timestep"""
        self.model.update()

    def verify_model_loaded(self) -> None:
        """Checks whether specified model is available."""
        if self.model_name in LOADED_MODELS:
            pass
        else:
            raise UserWarning(f"Defined model: {self.model} not loaded")


"""
Data assimilation methods
----------------------
"""



class ParticleFilter(BaseModel):
    """Implementation of a particle filter scheme to be applied to the :py:class:`Ensemble`.

    note:
        The :py:class:`ParticleFilter` is controlled by the :py:class:`Ensemble` and thus has no time reference itself.
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

        weights (np.ndarray): contains weights per ensemble member per prior modeled values [N x 1]

        resample_indices (np.ndarray): contains indices of particles that are resampled [N x 1]

        new_state_vectors (np.ndarray): updated state vector per ensemble member [N x len(z)]

    All are :obj:`None` by default


    """

    hyperparameters: dict = dict(like_sigma_weights=0.05, like_sigma_state_vector=0.0005)
    N: int
    obs: float | Any | None = None
    state_vectors: Any | None = None
    predictions: Any | None = None
    weights: Any | None = None
    resample_indices: Any | None = None
    new_state_vectors: Any | None = None


    def update(self):
        """Takes current state vectors of ensemble and returns updated state vectors ensemble
        """
        self.generate_weights()

        # TODO: Refactor to be more modular i.e. remove if/else

        # 1d for now: weights is N x 1
        if self.weights[0].size == 1:
            self.resample_indices = random.choices(population=np.arange(self.N), weights=self.weights, k=self.N)

            new_state_vectors = self.state_vectors.copy()[self.resample_indices]
            new_state_vectors_transpose = new_state_vectors.T # change to len(z) x N so in future you can vary sigma

            # for now just constant perturbation, can vary this hyperparameter
            like_sigma = self.hyperparameters['like_sigma_state_vector']
            for index, row in enumerate(new_state_vectors_transpose):
                row_with_noise = np.array([s + add_normal_noise(like_sigma)for s in row])
                new_state_vectors_transpose[index] = row_with_noise

            self.new_state_vectors = new_state_vectors_transpose.T # back to N x len(z) to be set correctly

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

            # for now just constant perturbation, can vary this hyperparameter
            like_sigma = self.hyperparameters['like_sigma_state_vector']
            for index, row in enumerate(new_state_vectors_transpose):
                row_with_noise = np.array([s + add_normal_noise(like_sigma) for s in row])
                new_state_vectors_transpose[index] = row_with_noise

            self.new_state_vectors = new_state_vectors_transpose.T  # back to N x len(z) to be set correctly



    def generate_weights(self):
        """Takes the ensemble and observations and returns the posterior

        Todo: Check if still correct function - iterated a lot through the notebooks

        """
        like_sigma = self.hyperparameters['like_sigma_weights']
        difference = (self.obs - self.predictions)
        unnormalised_log_weights = scipy.stats.norm.logpdf(difference, loc=0, scale=like_sigma)
        normalised_weights = np.exp(unnormalised_log_weights - scipy.special.logsumexp(unnormalised_log_weights))

        # for now fix to 2d
        # if normalised_weights[0].size > 1:
        #     normalised_weights = normalised_weights.mean(axis=1)
        self.weights = normalised_weights


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

    hyperparameters: dict = dict(like_sigma_state_vector=0.0005)
    N: int
    obs: Optional[float | None] = None
    state_vectors: Optional[Any | None] = None
    predictions: Optional[Any | None] = None
    new_state_vectors: Optional[Any | None] = None
    logger: list = [] # easier than using built in logger ?


    def update(self):
        """Takes current state vectors of ensemble and returns updated state vectors ensemble

        TODO: refactor to be more readable
        """

        # TODO: is obs are not float but array should be mXN, currently Nxm
        measurement_d = self.obs
        measurement_pertubation_matrix_E = np.array([add_normal_noise(self.hyperparameters['like_sigma_state_vector']) for x in range(self.N)])
        peturbed_measurements_D = measurement_d * np.ones(self.N).T + np.sqrt(
                                                                        self.N - 1) * measurement_pertubation_matrix_E
        predicted_measurements_Ypsilon = self.predictions
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
        D_tilde = np.matrix(peturbed_measurements_D - predicted_measurements_Ypsilon)

        W = S.T * np.linalg.inv(S * S.T + E * E.T) * D_tilde
        T = np.identity(self.N) + (W / np.sqrt(self.N - 1))

        self.new_state_vectors = np.array((prior_state_vector_Z * T).T) # back to N x len(z) to be set correctly



"""
Utility based functions
----------------------
"""

rng = np.random.default_rng() # Initiate a Random Number Generator
def add_normal_noise(like_sigma) -> float:
    """Normal (zero-mean) noise to be added to a state

    Args:
        like_sigma (float): scale parameter - pseudo variance & thus 'like'-sigma

    Returns:
        sample from normal distribution
    """
    return rng.normal(loc=0, scale=like_sigma)  # log normal so can't go to 0 ?





"""
Check methods - could also be static methods but as load_methods needs to be here for now refactor later? 
_____________
 
**keeps amount of boilerplate code lower and functions readable**

"""

LOADED_METHODS: dict[str, Any] = dict(
                                        PF=ParticleFilter,
                                        EnKF=EnsembleKalmanFilter,
                                     )
def validate_method(method):
    """"Checks uses supplied method to ensure """
    try:
        assert method in LOADED_METHODS
    except AssertionError:
        raise UserWarning(f"Method: {method} not loaded, ensure specified method is compatible")


def validity_initialize_input(model_name, forcing, setup_kwargs, state_vector_variables) -> None:
    """Checks user input to avoid confusion: if model_name is a list, all others must be too."""
    try:
        assert type(forcing) == list
        assert type(setup_kwargs) == list
        assert type(state_vector_variables) == list
    except AssertionError:
        raise UserWarning("forcing, setup_kwargs &"\
                         +"state_vector_variables should be list")
    try:
        assert len(model_name) == len(forcing)
        assert len(model_name) == len(setup_kwargs)
        assert len(model_name) == len(state_vector_variables)
    except AssertionError:
        raise UserWarning("Length of lists: model_name, forcing, setup_kwargs &"\
                         +"state_vector_variables should be the same length")