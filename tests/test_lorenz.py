# # test local version of the Lorenz models
import importlib.util
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# load 
DA = module_from_file("DA",r'../src/ewatercycle_DA/DA.py')
lorenz = module_from_file("lorenz",r'../src/ewatercycle_DA/lorenz.py')

def main():
    # ## Create ensemble
    n_particles = 5
    ensemble = DA.Ensemble(N=n_particles)
    ensemble.setup()
    
    # ### set up paths
    
    path = Path.cwd()
    forcing_path = path / "Forcing"
    observations_path = path / "Observations"
    figure_path = path / "Figures"
    
    # ## forcing
    
    experiment_start_date = "1997-10-01T00:00:00Z"
    experiment_end_date = "1997-10-10T00:00:00Z"
    
    lorenz_forcing = lorenz.LorenzForcing(start_time = experiment_start_date,
                                          end_time = experiment_end_date,
                                          directory = forcing_path,
                                          F = 8,
                                          dt=1e-3
                              )
    
    J = 40
    common_state = np.zeros(J)
    common_state[19] = 0.01
    store_noise = np.loadtxt(forcing_path / 'predefined_noise_1000.txt') 
    
    # add pre-run noise
    setup_kwargs_lst = []
    for index in range(n_particles):
        peturbed_state = common_state.copy()
        peturbed_state[5] += store_noise[index]
        setup_kwargs_lst.append({'J':J, 
                                'start_state':list(peturbed_state),
                                 })

    
    # this initializes the models for all ensemble members. 
    ensemble.initialize(model_name=["LorenzLocal"]*n_particles,
                        forcing=[lorenz_forcing]*n_particles,
                        setup_kwargs=setup_kwargs_lst) 
    
    
    # ## Run truth model
    
    truth_model = lorenz.LorenzLocal(forcing=lorenz_forcing)
    config, _ = truth_model.setup(J=J,
                            start_state=list(common_state))
    truth_model.initialize(config)
    
    n_timesteps = int(round((truth_model.end_time - truth_model.start_time) /  truth_model.time_step,0))
    
    output = pd.DataFrame(columns =['truth'])
    output_lst = []
    
    for _ in tqdm(range(n_timesteps)):   
        truth_model.update()
        output.loc[truth_model.time_as_datetime] = truth_model.get_value("state")[5]
        output_lst.append(truth_model.get_value("state"))
    truth_model.finalize()
    
    output.index.name = "time"
    current_time = str(datetime.now())[:-10].replace(":","_")
    ds_obs_dir = observations_path / f'truth_model_lorenz96_{current_time}.nc'
    # ds_obs = xr.Dataset(data_vars=output[['truth']])
    # ds_obs = xr.Dataset({'truth':xr.DataArray(data=pd.DataFrame(output_lst,index=output.index),dims=["time","x"])})
    ds_obs = xr.Dataset({'truth':xr.DataArray(data=pd.DataFrame(output,index=output.index),dims=["time","x"])})
    if not ds_obs_dir.exists():
        ds_obs.to_netcdf(ds_obs_dir)
    
    
    # ## setup DA method
    
    def H(Z):
        """Function which takes the state vector Z and returns part of that state. Returned should be same shape as data provided"""
        if len(Z) == J:
            return Z[5] 
        else: 
            raise SyntaxWarning(f"Length of statevector should be {J} but is {len(Z)}")
    
    ensemble.initialize_da_method(ensemble_method_name = "PF", 
                               hyper_parameters = {
                                                   'like_sigma_weights' : 0.05,
                                                   'like_sigma_state_vector' : 2.0,
                                                   },
                               state_vector_variables = ["state"],
                               observation_path = ds_obs_dir,
                               observed_variable_name = "truth",
                               measurement_operator= H                           
                                )
    
    ref_model = ensemble.ensemble_list[-1].model
    
    
    n_timesteps = int(round((ref_model.end_time - ref_model.start_time) /  ref_model.time_step,0))
    time = []
    lst_state_vector = []
    lst_Q = [] 
    for i in tqdm(range(n_timesteps)):    
        time.append(pd.Timestamp(ref_model.time_as_datetime.date()))
        assimilate = False
        ensemble.update(assimilate)
        lst_state_vector.append(ensemble.get_state_vector())
    
    # end model - IMPORTANT! when working with dockers
    ensemble.finalize()
    
    Q_m_arr = np.array(lst_Q).T
    state_vector_arr = np.array(lst_state_vector)
    data_vars = {}
    for i, name in enumerate(range(J)):
        storage_terms_i = xr.DataArray(state_vector_arr[:,:,i].T,
                                       name=name,
                                       dims=["EnsembleMember","time"],
                                      coords=[np.arange(n_particles),time[:len(state_vector_arr)]],
                                      attrs={"title": f"Lorenz states over time for {n_particles} particles ", 
                                               "history": f"states of lorenz model: ewatercycle DA",
                                            "description":"Moddeled values",
                                                 "units": "mm"})
        data_vars[name] = storage_terms_i
    
    ds_combined = xr.Dataset(data_vars,
                             attrs={"title": f"Lorenz model over time for {n_particles} particles ", 
                                    "history": f"states of lorenz model: ewatercycle DA",}
                              )

    
    tested_model_run_dir = observations_path / "tested_model_run.nc"
    ds_tested = xr.open_dataset(tested_model_run_dir)

    result = []
    x = 5
    for i in range(n_particles):
        result.append(sum(ds_combined[x].isel(EnsembleMember=i).values - ds_tested[f'{x}'].isel(EnsembleMember=i).values))
    assert np.isclose(sum(result), 0)
    
if __name__ == "__main__":
    main()