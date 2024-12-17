from .module import BaseModule

class BaseConfig(BaseModule):

    """
    a base config

    A config should inherent BaseConfig and specify the attributes and their types. 
    Otherwise this will be an empty config.
    """
    def save(self, path: str, **kwargs)-> str:
        super().save_module(path, **kwargs)

    
__all__ = ["BaseConfig"]

