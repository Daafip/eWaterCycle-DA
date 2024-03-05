from typing import Dict, Any, Type
from pathlib import Path

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


    def setup(self):
        """Creates a set of empty Ensemble member instances
        This allows further customisation: i.e. different models in one ensemble
        """
        if len(self.ensemble_list) != 0:
            self.ensemble_list = []
        for ensemble_member in range(self.N):
            self.ensemble_list.append(EnsembleMember())


    def initialize(self, model_name, forcing, setup_kwargs):
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
        # if all is the same for all members (to start with)
        if type(model_name) == str:
            for ensemble_member in self.ensemble_list:
                ensemble_member.model_name = model_name
                ensemble_member.forcing = forcing
                ensemble_member.setup_kwargs = setup_kwargs

                ensemble_member.verify_model_loaded()

                ensemble_member.setup()
                ensemble_member.initialize()

        # if a list is passed for more flexibility - could change in the furture
        elif type(model_name) == list and len(model_name) == self.N:
            for index_m, ensemble_member in enumerate(self.ensemble_list):
                ensemble_member.model_name = model_name[index_m]
                ensemble_member.forcing = forcing[index_m]
                ensemble_member.setup_kwargs = setup_kwargs[index_m]

                ensemble_member.verify_model_loaded()

                ensemble_member.setup()
                ensemble_member.initialize()
        else:
            raise SyntaxWarning(f"model should either string or list of string of length {self.N}")


    def finalize(self):
        """Runs finalize step for all members"""
        for ensemble_member in self.ensemble_list:
            ensemble_member.finalize()


class EnsembleMember(BaseModel):
    """Class containing ensemble members, meant to be called by the DA.Ensemble class"""

    model_name: str | None = None
    forcing: DefaultForcing | None = None
    setup_kwargs: dict | None = None
    model: Any | None = None
    config: Path | None = None

    def setup(self):
        self.model = loaded_models[self.model_name](forcing=self.forcing)
        self.config, _ = self.model.setup(**self.setup_kwargs)

    def initialize(self):
        self.model.initialize(self.config)

    def finalize(self):
        self.model.finalize()

    def verify_model_loaded(self):
        """"Check whether specified model is available."""
        if self.model_name in loaded_models:
            pass
        else:
            raise UserWarning(f"Defined model: {self.model} not loaded")
