"""Class of functions to integrate Data Assimilation into eWaterCycle

Note:
    assumes a 1D grid currently (e.g. in ``get_state_vector``) - not yet tested on distributed models.
"""
import numpy as np
import xarray as xr

import dask
from dask import delayed
import psutil

import warnings
import types

from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel

import ewatercycle
import ewatercycle.models
import ewatercycle.forcing
from ewatercycle.base.forcing import DefaultForcing


from ewatercycle_DA.utils import custom_make_cfg_dir
from ewatercycle_DA.data_assimilation_schemes.PF import ParticleFilter
from ewatercycle_DA.data_assimilation_schemes.EnKF import EnsembleKalmanFilter

from ewatercycle_DA.local_models.lorenz import LorenzLocal

LOADED_METHODS: dict[str, Any] = dict(
                                        PF=ParticleFilter,
                                        EnKF=EnsembleKalmanFilter,
                                     )

# saves users from encountering errors - change this to config file later?
KNOWN_WORKING_MODELS_DA: list[str] = ["HBV",
                                      "HBVLocal",
                                      "Lorenz",
                                      "LorenzLocal",
                                      "ParallelisationSleep"]

KNOWN_WORKING_MODELS_DA_HYDROLOGY: list[str] = ["HBV",
                                                "HBVLocal"]
TLAG_MAX = 100 # sets maximum lag possible (d)


def load_models(loaded_models) -> dict[str, Any]:
    """Loads models found in user install

    Note:
        To load other local models:
        .. code-block:: python

           from ewatercycle_DA import DA
           from model import LocalModel

           ensemble = DA.Ensemble(N=100)
           ensemble.loaded_models.update({'LocalModel':LocalModel})

        The LorenzLocal model is only added in this way as it is likely useful to benchmark DA schemes.

    """
    for model in ewatercycle.models.sources:
        loaded_models.update({model: ewatercycle.models.sources[model]})

    # append local models: can be done manually by user
    loaded_models.update({"LorenzLocal":LorenzLocal})
    return loaded_models

class Ensemble(BaseModel):
    """Class for running data assimilation in eWaterCycle

    Args:
        N : Number of ensemble members

        dask_config: Dictionary to pass to .. :py:`dask.config.set()`
                    see `dask docs <https://docs.dask.org/en/stable/scheduler-overview.html>`_
                    Will default to same number of workers and physical processors.
                    Use with care, too many workers will overload the infrastructure.

    Attributes:
        ensemble_method: method used for data assimilation

        ensemble_method_name: name of method used for data assimilation (needed for function specific)

        ensemble_list : list containing ensembleMembers

        observed_variable_name: Name of the observed value: often Q but could be anything

        measurement_operator: Function or list of Functions which maps the state vector to the measurement space:
            i.e. extracts the wanted value for comparison by the DA scheme from the state vector.   (Also known as H)

        observations: NetCDF file containing observations

        lst_models_name: list containing a set of all the model names: i.e. to run checks

        logger: list to debug issues, this isn't the best way to debug/log but works for me

        config_specific_storage: used by the config_specific_actions


    Note:
        Run ``setup`` and ``initialize`` before using other functions

    """

    N: int
    dask_config: dict = {"multiprocessing.context": "spawn",
                         'num_workers': psutil.cpu_count(logical=False)}

    ensemble_list: list = []
    ensemble_method: Any | None = None
    ensemble_method_name: str | None = None
    observed_variable_name: str | None = None
    measurement_operator: Any | list | None = None
    observations: Any | None = None
    lst_models_name: list = []
    logger: list = [] # logging proved too complex for now so just append to list XD
    config_specific_storage: Any | None = None

    loaded_models: dict[str, Any] = dict()
    loaded_models = load_models(loaded_models)

    def setup(self) -> None:
        """Creates a set of empty Ensemble member instances
        This allows further customisation: i.e. different models in one ensemble
        """
        if len(self.ensemble_list) != 0:
            self.ensemble_list = []
        for ensemble_member in range(self.N):
            self.ensemble_list.append(EnsembleMember())

    def generate_forcing(self,
             forcing_class: list[Any] | Any,
             list_forcing_args: list[dict],
                         ) -> list[Any]:
        """Generates forcing in parallel.

        Args:
            forcing_class (list | Any): (list of) forcing class instance(s). If only
                one is specified, will pass the different args to same class.

            list_forcing_args (list[dict]): list of arguments to be passed to the
                forcing class(es), should al be different.

        Example:
            e.g. to investigate the impact of varying a parameter:

            .. code-block:: python

                ensemble = DA.Ensemble(N=n_particles)
                ensemble.setup()

                experiment_start_date = "1997-08-01T00:00:00Z"
                experiment_end_date = "1999-03-01T00:00:00Z"
                HRU_id = 14138900

                forcing_args = []
                for i in range(1,11):
                    forcing_args.append(dict(
                        start_time = experiment_start_date,
                        end_time = experiment_end_date,
                        directory = forcing_path,
                        camels_file = f'0{HRU_id}_lump_cida_forcing_leap.txt',
                        alpha = 1.23 + (i/100)
                                             )
                                        )
                import ewatercycle.forcing
                ensemble.generate_forcing(ewatercycle.forcing.sources.HBVForcing,
                                              forcing_args
                                             )
            This returns 10 different forcing objects in a list which can be passed
            to ensemble.initialize
        """
        self.check_forcing_input(forcing_class, list_forcing_args)
        if type(forcing_class) == list:
            gathered_generate_forcing = (self.gather(
                *[self.generate_forcing_parallel(forcing_class[i],
                                                  list_forcing_args[i]
                                                  )
                                                  for i in range(self.N)]))
        else:
            gathered_generate_forcing = (self.gather(
                *[self.generate_forcing_parallel(forcing_class,
                                                  list_forcing_args[i]
                                                  )
                                                  for i in range(self.N)]))

        with dask.config.set(self.dask_config):
            forcing_objs = gathered_generate_forcing.compute()

        return forcing_objs


    def check_forcing_input(self, forcing_class, list_forcing_args) -> None:
        """Check user input for forcing is correct"""
        if type(forcing_class) == list:
            try:
                assert len(forcing_class) == len(list_forcing_args)
                assert len(forcing_class) == self.N
            except AssertionError:
                msg = ("Both supplied lists should be of same length & match "
                       "the ensemble length"
                       )
                raise RuntimeError(msg)
        else:
            try:
                assert len(list_forcing_args) == self.N
            except AssertionError:
                msg = "Both supplied list should be of same length as ensemble size"
                raise RuntimeError(msg)

    @staticmethod
    @delayed
    def generate_forcing_parallel(forcing, forcing_args):
        """Generate forcing given obj & args (with dask)."""
        forcing_obj = forcing(**forcing_args)
        return forcing_obj


    def initialize(self, model_name, forcing, setup_kwargs, custom_cfg_dir=True) -> None:
        """Takes empty Ensemble members and launches the model for given ensemble member

        Args:
            model_name (str | list): just takes the modl string name for now, change to more formal config file later?
                Should you pass a list here, you also need a list of forcing objects & vice versa.

            forcing (:obj:`ewatercycle.base.forcing.DefaultForcing` | :obj:`list`): object or list of objects to give to the model.
                Should you want to vary the forcing, you also need to pass a list of models.

            setup_kwargs (:obj:`dict` | :obj:`list`): kwargs dictionary which can be passed as `model.setup(**setup_kwargs)`.
                UserWarning: Ensure your model saves all kwargs to the config
                Should you want to vary initial parameters, again all should be a list

            custom_cfg_dir (Bool) by default will create a custom config dir similar to original.

            TODO: make it so custom config dir is emptied afterwards.

        Note:
            If you want to pass a list for any one variable, **all** others should be lists too of the same length.
        """

        # same for all members (to start with)
        if type(model_name) == str:
            for ensemble_member in self.ensemble_list:
                ensemble_member.model_name = model_name
                ensemble_member.forcing = forcing
                ensemble_member.setup_kwargs = setup_kwargs
                ensemble_member.loaded_models = self.loaded_models

        # more flexibility - could change in the future?
        elif type(model_name) == list and len(model_name) == self.N:
            validity_initialize_input(model_name, forcing, setup_kwargs)
            for index_m, ensemble_member in enumerate(self.ensemble_list):
                ensemble_member.model_name = model_name[index_m]
                ensemble_member.forcing = forcing[index_m]
                ensemble_member.setup_kwargs = setup_kwargs[index_m]
                ensemble_member.loaded_models = self.loaded_models
        else:
            raise SyntaxWarning(f"model should either string or list of string of length {self.N}")

        # setup & initialize - same in both cases - in parallel
        gathered_initialize = self.gather(*[self.initialize_parallel(self, i, custom_cfg_dir) for i in range(self.N)])

        with dask.config.set(self.dask_config):
            # starting too many dockers at once isn't great for the stability, limit to 1 for now
            lst_models_name = gathered_initialize.compute(num_workers=1)

        self.lst_models_name = list(set(lst_models_name))

    @staticmethod
    @delayed
    def gather(*args):
        return list(args)

    @staticmethod
    @delayed
    def initialize_parallel(ensemble, i, custom_cfg_dir):
        ensemble_member = ensemble.ensemble_list[i]
        ensemble_member.verify_model_loaded()
        if custom_cfg_dir:
            cfg_dir = custom_make_cfg_dir(ensemble_member.model_name, i)
            ensemble_member.cfg_dir = cfg_dir
        else:
            cfg_dir = None
        ensemble_member.setup(cfg_dir=cfg_dir)
        ensemble_member.initialize()
        return ensemble_member.model_name

    def initialize_da_method(self,
                          ensemble_method_name: str,
                          hyper_parameters: dict,
                          state_vector_variables: str | list,
                          observation_path: Path | None = None,
                          observed_variable_name: str | None = None,
                          measurement_operator: Any | list | None = None,
                          ):
        """Similar to initialize but specifically for the data assimilation method

        Args:
            ensemble_method_name (str): name of the data assimilation method for the ensemble


            hyper_parameters (dict): dictionary containing hyperparameters for the method, these will vary per method
                and thus are merely passed on

            state_vector_variables (Optional[str | :obj:`list[str]`]): can be set to 'all' for known parameters, this is
                highly model and scenario specific & should be implemented separately. Currently known to work for:

                 - ewatercycle-HBV
                 - ...

                Can be a set by passing a list containing strings of variable to include in the state vector.

                Changing to a subset allows you to do interesting things with ensembles: mainly limited to particle filters.

                For example giving half the particle filters more variables which vary than others - see what that does.

        Note:
            The following three are Keyword Args to make the code more flexible: when running the initialize_da_method
            to set up the ensemble normally *these are all needed*. They are separate as these aren't needed if DA is done
            on the fly.

        Keyword Args:
            observation_path (Path) = None: Path to a NetCDF file containing observations.
                Ensure the time dimension is of type :obj:`numpy.datetime64[ns]` in order to work well with
                 .. py:function:: `Ensemble.update`

            observed_variable_name (str) = None: Name of the observed value: often Q but could be anything

            measurement_operator (:obj:`function`| :obj:`list[functions]`) = None: if not specified: by default 'all' known parameters,
                can be a subset of all by passing a list containing strings of variable to include in the state vector.
                Should you want to vary initial parameters, again all should be a list

        Note:
            Assumed memory is large enough to hold observations in memory/lazy open with xarray
            Assumed memory is large enough to hold observations in memory/lazy open with xarray
        """
        validate_method(ensemble_method_name)

        self.ensemble_method = LOADED_METHODS[ensemble_method_name](N=self.N)
        self.ensemble_method_name = ensemble_method_name

        for hyper_param in hyper_parameters:
            self.ensemble_method.hyperparameters[hyper_param] = hyper_parameters[hyper_param]

        self.ensemble_method.N = self.N

        # TODO currently assumes state vector variables is the same for all ensemble members
        # TODO could also be list
        self.set_state_vector_variables(state_vector_variables)

        # only set if specified
        if not None in [observed_variable_name, observation_path, measurement_operator]:
            self.observed_variable_name = observed_variable_name
            self.observations = self.load_netcdf(observation_path,
                                                 observed_variable_name)
            self.measurement_operator = measurement_operator

    def set_state_vector_variables(self, state_vector_variables: str | list):
        """Sets state vector variables

        Called by `Ensemble.initialize_da_method`, but can also be called by user
        to set up get & set state vector if DA is not used.

        Args:
            state_vector_variables (Optional[str | :obj:`list[str]`]): can be set to
                'all' for known parameters, this is highly model and scenario specific
                 & should be implemented separately. Currently known to work for:

                 - ewatercycle-HBV
                 - ...

                Can be a set by passing a list containing strings of variable to
                include in the state vector.

                Changing to a subset allows you to do interesting things with ensembles:
                mainly limited to particle filters.

                For example giving half the particle filters more variables
                which vary than others - see what that does.

        """
        gathered_set_state_vect = self.gather(
            *[self.initialize_da_method_parallel(self, state_vector_variables, i)
                                                      for i in range(self.N)]
                                                )

        with dask.config.set(self.dask_config):
            gathered_set_state_vect.compute()


    @staticmethod
    @delayed
    def initialize_da_method_parallel(ensemble, state_vector_variables, i):
        ensemble_member = ensemble.ensemble_list[i]
        ensemble_member.state_vector_variables = state_vector_variables
        ensemble_member.set_state_vector_variable()


    def finalize(self, remove_config=True) -> None:
        """Runs finalize step for all members

        Optional Arg:
            remove_config (Bool) = True: removes created config paths on finalizing by default.
                set to false in case you wish to keep.

        """
        gathered_finalize = (self.gather(*[self.finalize_parallel(self, i, remove_config) for i in range(self.N)]))

        with dask.config.set(self.dask_config):
            gathered_finalize.compute()

    # TODO: think if this is a good idea
    @staticmethod
    @delayed
    def finalize_parallel(ensemble, i, remove_config):
        ensemble_member = ensemble.ensemble_list[i]
        ensemble_member.finalize(remove_config)

    def update(self, assimilate=False) -> None:
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

        gathered_update = (self.gather(*[self.update_parallel(self, i) for i in range(self.N)]))

        with dask.config.set(self.dask_config):
            gathered_update.compute()

        if assimilate:
            if not all(model_name in KNOWN_WORKING_MODELS_DA for model_name in self.lst_models_name):
                raise RuntimeWarning(f'Not all models specified {self.lst_models_name} are known to work with' \
                                    +'Data Assimilation. Either specify model that does work or submit a PR to add it.')
            # get observations
            current_time = np.datetime64(self.ensemble_list[0].model.time_as_datetime)
            current_obs = self.observations.sel(time=current_time, method="nearest").values


            self.assimilate(ensemble_method_name=self.ensemble_method_name,
                            obs=current_obs,
                            measurement_operator = self.measurement_operator,
                            hyper_parameters = self.ensemble_method.hyperparameters,
                            state_vector_variables = None, # maybe fix later? - currently don't have access
                            )


    @staticmethod
    @delayed
    def update_parallel(ensemble, i):
        ensemble_member = ensemble.ensemble_list[i]
        ensemble_member.update()

    def assimilate(self,
                   ensemble_method_name: str,
                   obs: np.ndarray,
                   measurement_operator,
                   hyper_parameters: dict,
                   state_vector_variables: str | list | None,
                   ):

        """ Similar to calling .. py:function:: Ensemble.update(assimilate=True)
        Intended for advanced users!
        The assimilate class aims to make on the fly data assimilation possible.
        You only need to define which method, observations and H operator you wish to use.
        This however requires more know-how of the situation,

        Args:
            ensemble_method_name (str): name of the data assimilation method for the ensemble

            obs (np.ndarray): array of observations for given timestep.

            measurement_operator (:obj:`function`| :obj:`list[functions]`): maps

            hyper_parameters (dict): dictionary of hyperparameters to set to DA method

            state_vector_variables (str | :obj:`list[str]`| None): can be set to 'all' for known parameters, this is
                highly model and scenario specific & should be implemented separately. Currently known to work for:

                 - ewatercycle-HBV
                 - ...

                Can be a set by passing a list containing strings of variable to include in the state vector.

                Changing to a subset allows you to do interesting things with ensembles: mainly limited to particle filters.

                For example giving half the particle filters more variables which vary than others - see what that does.

                set to None is called from .. py:function: Ensemble.update(assimilate=true)

        """
        # if on the fly da: we need to initialize here:
        if self.ensemble_method is None:
            self.initialize_da_method(ensemble_method_name=ensemble_method_name,
                                      hyper_parameters=hyper_parameters,
                                      state_vector_variables = state_vector_variables)

        self.ensemble_method.state_vectors = self.get_state_vector()

        self.ensemble_method.predictions = self.get_predicted_values(measurement_operator)

        self.ensemble_method.obs = obs

        self.ensemble_method.update()

        self.remove_negative()

        self.config_specific_actions(pre_set_state=True)

        self.set_state_vector(self.ensemble_method.new_state_vectors)

        self.config_specific_actions(pre_set_state=False)


    def get_predicted_values(self, measurement_operator) -> np.ndarray:
        """"Loops over the state vectors and applies specified measurement operator to obtain predicted value"""
        predicted_values = []
        if type(measurement_operator) == list:
            # if a list is passed, it's a list of operators per ensemble member
            for index, ensemble_state_vector in enumerate(self.ensemble_method.state_vectors):
                predicted_values.append(measurement_operator[index](ensemble_state_vector))

        elif type(measurement_operator) == types.FunctionType:
            # if a just a function is passed, apply same to all
            for ensemble_state_vector in self.ensemble_method.state_vectors:
                predicted_values.append(measurement_operator(ensemble_state_vector))
        else:
            raise RuntimeError(f"Invalid type {measurement_operator}, should be either list of function but is ")

        return np.vstack(predicted_values)

    def remove_negative(self):
        """if only one model is loaded & hydrological: sets negative numbers to positive
           Other models such as the lorenz model can be negative"""
        # in future may be interesting to load multiple types of hydrological models in one ensemble
        # for not not implemented
        if len(self.lst_models_name) == 1 and self.lst_models_name[0] in KNOWN_WORKING_MODELS_DA_HYDROLOGY:
                # set any values below 0 to small
                self.ensemble_method.new_state_vectors[self.ensemble_method.new_state_vectors < 0] = 1e-6

        elif len(self.lst_models_name) == 1:
            warnings.warn(f"Model {self.lst_models_name[0]} loaded not flagged as hydrological" \
                          +"no negative values removed", category=UserWarning)
        else:
            warnings.warn("More than 1 model type loaded, no negative values removed",category=RuntimeWarning)

    def config_specific_actions(self, pre_set_state):
        """Function for actions which are specific to a combination of model with method.

            Note:
                Be specific when these are used to only occur when wanted

            *#1: PF & HBV*:
                Particle filters replace the full particle: thus the lag function also needs to be copied.

                If only HBV models are implemented with PF this will be updates

                if HBV and other models are implemented, this will present a RuntimeWarning.

                If other models are implemented with PF, nothing should happen, just a UserWarning so you're aware.


        """
        # TODO: Refactor so BMI handles this
        #1
        if self.ensemble_method_name == "PF" and self.ensemble_method.resample:
            # in particle filter the whole particle needs to be copied
            # when dealing with lag this is difficult as we don't want it in the regular state vector
            # only deal with this if resampled
            hbv_model = "HBV" in self.lst_models_name
            hbv_local_model = "HBVLocal" in self.lst_models_name
            if (hbv_model or hbv_local_model) and len(self.lst_models_name) == 1:
                if pre_set_state:
                    # first get the memory vectors for all ensemble members
                    lag_vector_arr = np.zeros((len(self.ensemble_list),TLAG_MAX))
                    for index, ensemble_member in enumerate(self.ensemble_list):
                        t_lag = int(ensemble_member.get_value("Tlag")[0])
                        old_t_lag = np.array([ensemble_member.get_value(f"memory_vector{i}") for i in range(t_lag)]).flatten()
                        lag_vector_arr[index,:t_lag] = old_t_lag

                    self.config_specific_storage = lag_vector_arr

                else:
                    lag_vector_arr = self.config_specific_storage
                    # resample so has the correct state
                    # TODO consider adding noise ?
                    new_lag_vector_lst = lag_vector_arr[self.ensemble_method.resample_indices]

                    for index, ensembleMember in enumerate(self.ensemble_list):
                        new_t_lag = ensembleMember.get_value(f"Tlag")
                        [ensembleMember.set_value(f"memory_vector{mem_index}", np.array([new_lag_vector_lst[index, mem_index]])) for mem_index in range(int(new_t_lag))]

            elif hbv_model or hbv_local_model:
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
        shape_data = ref_model.get_value(var_name).shape[0]

        output_array = np.zeros((self.N, shape_data))

        self.logger.append(f'{output_array.shape}')

        gathered_get_value = (self.gather(*[self.get_value_parallel(self,var_name, i) for i in range(self.N)]))

        with dask.config.set(self.dask_config):
            get_value_lst = gathered_get_value.compute()

        output_array = np.vstack(get_value_lst)
        self.logger.append(f'{output_array.shape}')
        return output_array

    @staticmethod
    @delayed
    def get_value_parallel(ensemble, var_name, i):
        ensemble_member = ensemble.ensemble_list[i]
        return ensemble_member.model.get_value(var_name)

    def get_state_vector(self) -> np.ndarray:
        """Gets current value of whole ensemble for specified state vector
            Note:
                Assumes 1d array? although :obj:`np.vstack` does work for 2d arrays
        """
        gathered_get_state_vector = (self.gather(*[self.get_state_vector_parallel(self, i) for i in range(self.N)]))

        with dask.config.set(self.dask_config):
            output_lst = gathered_get_state_vector.compute()

        return np.vstack(output_lst) # N x len(z)

    @staticmethod
    @delayed
    def get_state_vector_parallel(ensemble, i):
        ensemble_member = ensemble.ensemble_list[i]
        return ensemble_member.get_state_vector()

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        """Sets current value of whole ensemble for given variable
            args:
                src (np.ndarray): size = number of ensemble members x 1 [N x 1]
        """
        gathered_set_value = (self.gather(*[self.set_value_parallel(self, var_name, src[i], i) for i in range(self.N)]))

        with dask.config.set(self.dask_config):
            gathered_set_value.compute()

    @staticmethod
    @delayed
    def set_value_parallel(ensemble, var_name, src_i, i):
        ensemble_member = ensemble.ensemble_list[i]
        return ensemble_member.model.set_value(var_name,src_i)

    def set_state_vector(self, src: np.ndarray) -> None:
        """Sets current value of whole ensemble for specified state vector

            args:
                src (np.ndarray): size = number of ensemble members x number of states in state vector [N x len(z)]
                    src[0] should return the state vector for the first value
        """
        gathered_set_state_vector = (self.gather(*[self.set_state_vector_parallel(self,src[i], i) for i in range(self.N)]))

        with dask.config.set(self.dask_config):
            gathered_set_state_vector.compute()


    @staticmethod
    @delayed
    def set_state_vector_parallel(ensemble, src_i, i):
        ensemble_member = ensemble.ensemble_list[i]
        return ensemble_member.set_state_vector(src_i)

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

        state_vector_variables (Optional[str | :obj:`list[str]`]): can be set to 'all' for known parameters, this is
            highly model and scenario specific & should be implemented separately. Currently known to work for:
                 - ewatercycle-HBV
                 - ...
            Can be a set by passing a list containing strings of variable to include in the state vector.
            Changing to a subset allows you to do interesting things with ensembles: mainly limited to particle filters.
            For example giving half the particle filters more variables which vary than others - see what that does.
            TODO: refactor to be more on the fly


    Attributes:
        model (:obj:`ewatercycle.base.model`): instance of eWaterCycle model to be used.
            Must be defined in ``loaded_models`` dictionary in this file which is a safeguard against misuse.

        config (:obj:`Path`): path to config file for the model which the EnsembleMember contains.

        state_vector (:obj:`np.ndarray`): numpy array containing last states which were gotten

        variable_names (list[str]): list of string containing the variables in the state vector.

        loaded_models (dict[str, Any]): dictionary containing model names and their corresponding instances

    """

    model_name: str | None = None
    forcing: DefaultForcing | None = None
    setup_kwargs: dict | None = None

    state_vector_variables: Optional[str | list] = None

    model: Any | None = None
    config: str | None = None
    cfg_dir: Path | None = None
    state_vector: Any | None = None
    variable_names: list[str] | None = None
    loaded_models: dict[str, Any] = dict()

    def setup(self, cfg_dir) -> None:
        """Setups the model provided with forcing and kwargs. Set the config file"""
        self.model = self.loaded_models[self.model_name](forcing=self.forcing)
        self.config, _ = self.model.setup(cfg_dir=cfg_dir,**self.setup_kwargs)

    def initialize(self) -> None:
        """Initializes the model with the config file generated in setup"""
        self.model.initialize(self.config)

    def set_state_vector_variable(self):
        """"Set the list of  variables required to obtain the state vector"""
        if self.state_vector_variables is None:
            raise UserWarning(f'State_vector_variables: {self.state_vector_variables}'
                              +"Must be 'all' or list[str] containing wanted variables.")

        elif self.state_vector_variables == "all":
            if self.model_name == "HBV" or self.model_name == "HBVLocal":
                self.variable_names = list(dict(self.model.parameters).keys()) \
                                      + list(dict(self.model.states).keys())   \
                                      + ["Q"]
            # elif self.model == "..."
            else:
                raise RuntimeWarning(f"Default 'all' is not specified for {self.model_name}" \
                                     + "Please pass a list of variable or submit PR.")
                # also change the documentation initialize_da_method

        # TODO more elegant type checking availible
        elif type(self.state_vector_variables) == list and type(self.state_vector_variables[0]) == str:
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
        if self.variable_names is None:
            msg = (
                'First set variable names through `ensemble.initialize_da_method`'
                'if DA is not used, then through `ensemble.set_state_vector_variables`'
            )
            raise UserWarning(msg)

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
        self.model.set_value(var_name, np.array([src]))

    def set_state_vector(self,src: np.ndarray) -> None:
        """Sets current state vector of ensemble member
        TODO: check this
        Note: assumes a 1D grid currently as ``state_vector`` is 1D array.
        """
        for v_index, var_name in enumerate(self.variable_names):
            self.set_value(var_name, src[v_index])

    def finalize(self, remove_config) -> None:
        """"Finalizes the model: closing containers etc. if necessary"""
        self.model.finalize()
        if remove_config:
            try:
                Path(self.config).unlink()
            except FileNotFoundError:
                warnings.warn(message=f"{self.config} not found", category=UserWarning)
            try:
                self.cfg_dir.rmdir()
            except FileNotFoundError:
                warnings.warn(f"{self.cfg_dir} not found", category=UserWarning)

    def update(self) -> None:
        """Updates the model to the next timestep"""
        self.model.update()

    def verify_model_loaded(self) -> None:
        """Checks whether specified model is available."""
        if self.model_name in self.loaded_models:
            pass
        else:
            raise UserWarning(f"Defined model: {self.model} not loaded")

"""
Check methods - could also be static methods but as load_methods needs to be here for now refactor later? 
_____________
 
**keeps amount of boilerplate code lower and functions readable**

"""

def validate_method(method):
    """"Checks uses supplied method to ensure """
    try:
        assert method in LOADED_METHODS
    except AssertionError:
        raise UserWarning(f"Method: {method} not loaded, ensure specified method is compatible")


def validity_initialize_input(model_name, forcing, setup_kwargs) -> None:
    """Checks user input to avoid confusion: if model_name is a list, all others must be too."""
    try:
        assert type(forcing) == list
        assert type(setup_kwargs) == list
    except AssertionError:
        raise UserWarning("forcing & setup_kwargs should be list")
    try:
        assert len(model_name) == len(forcing)
        assert len(model_name) == len(setup_kwargs)
    except AssertionError:
        raise UserWarning("Length of lists: model_name, forcing & setup_kwargs should be the same length")


