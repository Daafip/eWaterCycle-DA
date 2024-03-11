"""Class of functions to integrate Data Assimilation into eWaterCycle

Note:
    assumes a 1D grid currently (e.g. in ``get_state_vector``) - not yet tested on distributed models.
"""

import scipy
import numpy as np
import xarray as xr
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel
from ewatercycle.base.forcing import DefaultForcing
import ewatercycle
from ewatercycle.models import HBV
from xarray import DataArray

loaded_models: dict[str, Any] = dict(
                                        HBV=ewatercycle.models.HBV,
                                     )


class Ensemble(BaseModel):
    """Class for running data assimilation in eWaterCycle

    Args:
        N : Number of ensemble members

        location : Where the model is run, by default local - change to remote later

    Attributes:
        ensemble_method: method used for data assimilation

        ensemble_list : list containing ensembleMembers

        observed_variable_name: Name of the observed value: often Q but could be anything

        observations: NetCDF file containing observations


    Note:
        Run ``setup`` and ``initialize`` before using other functions

    """

    N: int
    location: str = "local"

    ensemble_list: list = []
    ensemble_method: Optional[Any | None] = None
    observed_variable_name: str | None = None
    observations: Any | None = None


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

    def initialize_method(self, ensemble_method_name: str, observation_path: Path, observed_variable_name: str, hyper_parameters: dict):
        """Similar to initialize but specifically for the data assimilation method

        Args:
            ensemble_method_name (str): name of the data assimilation method for the ensemble

            observation_path (Path): Path to a NetCDF file containing observations

            observed_variable_name (str): Name of the observed value: often Q_m but could be anything

            hyper_parameters (dict): dictionary containing hyperparameters for the method, these will vary per method
                and thus are merely passed on

        Note:
            Assumed memory is large enough to hold observations in memory/lazy open with xarrays
        """
        validate_method(ensemble_method_name)
        self.ensemble_method = loaded_methods[ensemble_method_name]()
        self.observed_variable_name = observed_variable_name
        self.observations = load_netcdf(observation_path, observed_variable_name)
        for hyper_param in hyper_parameters:
            self.ensemble_method.hyperparameters[hyper_param] = hyper_parameters[hyper_param]


    def finalize(self) -> None:
        """Runs finalize step for all members"""
        for ensemble_member in self.ensemble_list:
            ensemble_member.finalize()

    def update(self) -> None:
        """Updates model for all members.
         Object-oriented programing, thus the state vector lives `in` the ensemble members

         Todo: think about assimilation windows not being every timestep
         """
        for ensemble_member in self.ensemble_list:
            # prior = ensemble_member.get_state_vector() ## in case you need it before updating
            ensemble_member.update()
            ensemble_member.get_state_vector()

        # here reference to the observations.
        current_time = self.ensemble_list[0].model.time
        current_obs = self.observations.sel(time=current_time)

        # TODO: continue here
        # TODO: continue here
        # TODO: continue here

    def get_value(self, var_name: str) -> np.ndarray:
        """Gets current value of whole ensemble for given variable"""
        output_array = np.zeros(self.N)
        for i, ensemble_member in enumerate(self.ensemble_list):
            output_array[i] = ensemble_member.model.get_value(var_name)[0]
        return output_array

    def get_state_vector(self) -> np.ndarray:
        """Gets current value of whole ensemble for specified state vector
            Note:
                Assumes 1d array? although `np.vstack` does work for 2d arrays
        """
        output_lst = []
        for ensemble_member in self.ensemble_list:
            output_lst.append(ensemble_member.get_state_vector())
        return np.vstack(output_lst)

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        """Sets current value of whole ensemble for given variable
            args:
                src (np.ndarray): size = number of ensemble members x 1
        """
        for i, ensemble_member in enumerate(self.ensemble_list):
            ensemble_member.model.set_value(var_name, src[i])

    def set_state_vector(self, src: np.ndarray) -> None:
        """Sets current value of whole ensemble for specified state vector

            args:
                src (np.ndarray): size = number of ensemble members x number of states in state vector
                    src[0] should return the state vector for the first value
        """
        for i, ensemble_member in enumerate(self.ensemble_list):
            ensemble_member.model.set_state_vector(src[i])


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
        self.model = loaded_models[self.model_name](forcing=self.forcing)
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
        Note: assumes a 1D grid currently as ``state_vector`` is 1D array.
        """
        self.state_vector = np.zeros(len(self.variable_names))
        for v_index, var_name in enumerate(self.variable_names):
            self.state_vector[v_index] = self.get_value(var_name)

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
        """Updates the model to the next timestep: could be useful to have a"""
        self.model.update()

    def verify_model_loaded(self) -> None:
        """Checks whether specified model is available."""
        if self.model_name in loaded_models:
            pass
        else:
            raise UserWarning(f"Defined model: {self.model} not loaded")


class ParticleFilter(BaseModel):
    """Implementation of a particle filter scheme to be applied to the :py:class:`Ensemble`.

    note:
        The :py:class:`ParticleFilter` is controlled by the :py:class:`Ensemble` and thus has no time reference itself.
        No DA method should need to know where in time it is (for now).

    Attributes:
        obs (float | None): observation value of the current model timestep, initially None if not set.
        like_sigma_weights (float): scale parameter - pseudo variance & thus 'like'-sigma used for weight function




    """

    obs: Optional[float | None] = None
    hyperparameters: dict =  dict(like_sigma_weights=0.05)

    def generate_weights(self, prior, obs) -> np.ndarray:
        """Takes the ensemble and observations and returns the posterior
        Args:
            prior (np.ndarray): modeled values for different particles
            obs (float): observed value

        Returns:
            normalised_weights (np.ndarray): weights of normalised likelihood - closer to observed is more likely

        Todo: Check if still correct function - iterated a lot through the notebooks

        """
        like_sigma = self.hyperparameters['like_sigma_weights']
        difference = (obs - prior)
        unnormalised_log_weights = scipy.stats.norm.logpdf(difference, loc=0, scale=like_sigma)
        normalised_weights = np.exp(unnormalised_log_weights - scipy.special.logsumexp(unnormalised_log_weights))
        return normalised_weights


class EnsembleKalmanFilter(BaseModel):
    """Implementation of a particle filter scheme to be applied to the :py:class:`Ensemble`.

      note:
          The :py:class:`ParticleFilter` is controlled by the :py:class:`Ensemble` and thus has no time reference itself.
          No DA method should need to know where in time it is (for now).

      Attributes:
          obs (float | None): observation value of the current model timestep, initially None if not set.

      """
    obs: float | None = None
    pass

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


def load_netcdf(observation_path: Path, observed_variable_name: str) -> xr.DataArray:
    """Load the observation data file supplied by user"""
    data = xr.open_dataset(observation_path)
    try:
        assert "time" in data.dims
        assert observed_variable_name in data.data_vars
    except AssertionError:
        raise UserWarning(f"Time or {observed_variable_name} not present in NetCDF file presented")
    return data[observed_variable_name]


"""
Check methods
_____________
 
**keeps amount of boilerplate code lower and functions readable**

"""

loaded_methods: dict[str, Any] = dict(
                                        PF=ParticleFilter,
                                        EnFK=EnsembleKalmanFilter,
                                     )
def validate_method(method):
    """"Checks uses supplied method to ensure """
    try:
        assert method in loaded_methods
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