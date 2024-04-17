# eWaterCycle-DA
<!-- [![Python package](https://github.com/Daafip/ewatercycle-hbv/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Daafip/ewatercycle-hbv/actions/workflows/test.yml) -->
[![Documentation Status](https://readthedocs.org/projects/ewatercycle-da/badge/?version=latest)](https://ewatercycle-da.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/ewatercycle-DA)](https://pypi.org/project/ewatercycle-DA/)
[![github license badge](https://img.shields.io/github/license/Daafip/ewatercycle-DA)](https://github.com/Daafip/ewatercycle-DA)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8B-yellow)](https://fair-software.eu)




Code to run Data Assimilation with hydrological models on the [eWaterCycle](https://github.com/eWaterCycle/ewatercycle) platform. 

## Installation
Install this package alongside your [eWaterCycle installation](https://github.com/eWaterCycle/ewatercycle/blob/main/README.md#install)

```console
pip install ewatercycle-DA
```

Then DA becomes available to be used in eWaterCycle

```python
from ewatercycle_DA import DA

```


## docs
Documentation can be found [here](https://ewatercycle-da.readthedocs.io/en/latest/)

## Changelog
Changelog can be found in [CHANGELOG.md](https://github.com/Daafip/ewatercycle-da/blob/main/CHANGELOG.md) on GitHub. 

## Quick Usage overiew
_(maybe migrate this to docs?)_

Can be used with or without assimilating, this will run 10 versions of the same model.
By varying the setup_kwargs you can vary the model run itself.

### Without assimilating
```py

HBVForcing = ...

ensemble = DA.Ensemble(N=10)
ensemble.setup()

ensemble.initialize(model_name="HBV",
                   forcing=HBVForcing,
                   setup_kwargs={'parameters':'7.6,0.5,460,3.8,0.19,1.3,0.082,0.0061',
                                 'initial_storage':'0,100,0,5'}
                    )

ref_model = ensemble.ensemble_list[0].model

lst_Q = []
while ref_model.time < ref_model.end_time:
    ensemble.update(assimilate=False)
    lst_Q.append(ensemble.get_value("Q"))
```
_For running HBV see seperate [docs](https://github.com/Daafip/ewatercycle-hbv)_
### With assimilating

```py

...
ref_model = ...
#... same as above just add two more definitions
def H(Z):
    """returns discharge which is the last value on the state vector for HBV"""
    return Z[-1] 

ds_obs_dir = ...
ensemble.initialize_da_method(ensemble_method_name = "PF", 
                              hyper_parameters = {
                                               'like_sigma_weights' : 0.05,
                                               'like_sigma_state_vector' : 0.01,
                                                 },
                              state_vector_variables = "all", 
                              # the next three are keyword arguments but are needed:
                              observation_path = ds_obs_dir,
                              observed_variable_name = "Q",
                              measurement_operator = H, 
                           
                            )
lst_Q = []
while ref_model.time < ref_model.end_time:
    ensemble.update(assimilate=True)
    lst_Q.append(ensemble.get_value("Q"))

```