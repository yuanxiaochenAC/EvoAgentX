import os 
import re 
from typing import Union, Any, List, Set

from ..core.logging import logger

def make_parent_folder(path: str):

    dir_folder = os.path.dirname(path)
    if not os.path.exists(dir_folder):
        logger.info(f"creating folder {dir_folder} ...")
        os.makedirs(dir_folder, exist_ok=True)

def safe_remove(data: Union[List[Any], Set[Any]], remove_value: Any):
    try:
        data.remove(remove_value)
    except ValueError:
        pass

def generate_dynamic_class_name(base_name: str) -> str:

    base_name = base_name.strip()
    
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s]', '', base_name)
    components = cleaned_name.split()
    class_name = ''.join(x.capitalize() for x in components)

    return class_name if class_name else 'DefaultClassName'
