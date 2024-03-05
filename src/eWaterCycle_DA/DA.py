from typing import Dict, Any, Type
from pathlib import Path

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PrivateAttr,
    model_validator,
)

from ewatercycle.base.forcing import DefaultForcing
import ewatercycle
from ewatercycle.base.model import eWaterCycleModel
from ewatercycle.models import HBV

loaded_models: dict[str, Any] = {"HBV": ewatercycle.models.HBV}
class Ensemble(BaseModel):
    """Class for running data assimilation in eWaterCycle

    Attributes:
        N: number of ensemble members
        location: Where the model is run, by default local - change to remote later
        ensemble: list of ensembleMembers
    """

    N: int
    location: str = "local"
    ensemble_list: list = []

    def setup(self) -> None:
        """Creates a set of empty Ensemble member instances
        This allows further customisation: i.e. different models in one ensemble
        """
        if len(self.ensemble_list) != 0:
            self.ensemble_list = []
        for ensemble_member in range(self.N):
            self.ensemble_list.append(EnsembleMember())

    def initialize(self, model_name, forcing, setup_kwargs) -> None:
        """"Takes empty Ensemble members and launches the model for given ensemble member
        Args:
            model_name: str | list: just takes the model string name for now, change to more formal config file later?
                               Should you pass a list here, you also need a list of forcing objects & vice versa.

            forcing: ewatercycle.base.forcing | list: object or list of objects to give to the model.
                                                      Should you want to vary the forcing, you also need to pass a list
                                                      of models.

            setup_kwargs: dict | list: kwargs dictionary which can be passed as `model.setup(**setup_kwargs)`.
                        UserWarning: Ensure your model saves all kwargs to the config
                        Should you want to vary initial parameters, again all should be a list
        """
        # same for all members (to start with)
        if type(model_name) == str:
            for ensemble_member in self.ensemble_list:
                ensemble_member.model_name = model_name
                ensemble_member.forcing = forcing
                ensemble_member.setup_kwargs = setup_kwargs
        # more flexibility - could change in the future?
        elif type(model_name) == list and len(model_name) == self.N:
            for index_m, ensemble_member in enumerate(self.ensemble_list):
                ensemble_member.model_name = model_name[index_m]
                ensemble_member.forcing = forcing[index_m]
                ensemble_member.setup_kwargs = setup_kwargs[index_m]
        else:
            raise SyntaxWarning(f"model should either string or list of string of length {self.N}")

        # setup & initialize - same in both cases
        for ensemble_member in self.ensemble_list:
            ensemble_member.verify_model_loaded()
            ensemble_member.setup()
            ensemble_member.initialize()

    def finalize(self) -> None:
        """Runs finalize step for all members"""
        for ensemble_member in self.ensemble_list:
            ensemble_member.finalize()

    def update(self) -> None:
        """Updates model for all members"""
        for ensemble_member in self.ensemble_list:
            ensemble_member.update()

    def get_value(self, var_name: str) -> np.ndarray:
        output_array = np.zeros(self.N)
        for i, ensemble_member in enumerate(self.ensemble_list):
            output_array[i] = ensemble_member.model.get_value(var_name)[0]
        return output_array

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        for i, ensemble_member in enumerate(self.ensemble_list):
            ensemble_member.model.set_value(var_name, src[i])


class EnsembleMember(BaseModel):
    """Class containing ensemble members, meant to be called by the DA.Ensemble class"""

    model_name: str | None = None
    forcing: DefaultForcing | None = None
    setup_kwargs: dict | None = None
    model: Any | None = None
    config: Path | None = None

    def setup(self) -> None:
        self.model = loaded_models[self.model_name](forcing=self.forcing)
        self.config, _ = self.model.setup(**self.setup_kwargs)

    def initialize(self) -> None:
        self.model.initialize(self.config)

    def get_value(self, var_name: str) -> np.ndarray:
        return self.model.get_value(var_name)

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        self.model.set_value(var_name, src)

    # TODO: reevaluate if this is even worth it? - hind sight: probably
    def finalize(self) -> None:
        self.model.finalize()

    def update(self) -> None:
        self.model.update()

    def verify_model_loaded(self) -> None:
        """"Check whether specified model is available."""
        if self.model_name in loaded_models:
            pass
        else:
            raise UserWarning(f"Defined model: {self.model} not loaded")
