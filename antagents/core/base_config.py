from pydantic import BaseModel
from .module import BaseModule

class BaseConfig(BaseModule):

    """
    a base config

    A config should inherent BaseConfig and specify the attributes and their types. 
    Otherwise this will be an empty config.
    """
    def save(self, path: str, **kwargs)-> str:
        super().save_module(path, **kwargs)

    def get_set_params(self):
        explicitly_set_fields = {field: getattr(self, field) for field in self.__fields_set__}
        if self.kwargs:
            explicitly_set_fields.update(self.kwargs)
        return explicitly_set_fields


class Parameter(BaseModel):
    
    name: str
    type: str 
    description: str 

    
__all__ = ["BaseConfig"]

