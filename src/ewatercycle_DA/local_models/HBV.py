# for local mode
from typing import Type
from ewatercycle.base.model import LocalModel
from ewatercycle_HBV.model import HBVMethods
from HBV import HBV as HBV_bmi
from bmipy import Bmi

# TODO: Local models shouldn't really be here
class HBVLocal(LocalModel, HBVMethods):
    """The HBV eWaterCycle model, with the local BMI."""
    bmi_class: Type[Bmi] = HBV_bmi