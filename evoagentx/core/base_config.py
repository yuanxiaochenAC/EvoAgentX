# from pydantic import BaseModel
from typing import List
from .module import BaseModule

class BaseConfig(BaseModule):

    """
    a base config

    A config should inherent BaseConfig and specify the attributes and their types. 
    Otherwise this will be an empty config.
    """
    def save(self, path: str, **kwargs)-> str:
        super().save_module(path, **kwargs)

    def get_config_params(self):
        config_params = list(type(self).model_fields.keys())
        config_params.remove("class_name")
        return config_params

    def get_set_params(self, ignore: List[str] = []):
        explicitly_set_fields = {field: getattr(self, field) for field in self.__fields_set__}
        if self.kwargs:
            explicitly_set_fields.update(self.kwargs)
        for field in ignore:
            explicitly_set_fields.pop(field, None)
        return explicitly_set_fields


class Parameter(BaseModule):
    name: str
    type: str 
    description: str 
