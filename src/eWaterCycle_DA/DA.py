from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PrivateAttr,
    model_validator,
)

class DA(BaseModel):
    """"Class for running data assimilation in eWaterCycle"""