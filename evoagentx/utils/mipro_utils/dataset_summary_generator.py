import re

from .settings import settings
from evoagentx.utils.mipro_utils.utils import (
    strip_prefix,
    order_input_keys_in_string
)
from evoagentx.agents import CustomizeAgent

dataset_descriptor = CustomizeAgent(
    name="DatasetDescriptor",
    description="A simple agent that describes a dataset",
    prompt="""
    Given several examples from a dataset please write observations about trends that hold for most or all of the samples. 
    Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. 
    It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative
    """,
    inputs = [
        {"name": "examples", "type": "list", "desc": "Sample data points from the dataset"}
    ],
    outputs = [
        {"name": "observations", "type": "str", "desc": "Somethings that holds true for most or all of the data you observed"}
    ],
    llm_config=settings.lm,
    parse_mode=str,
)



def create_dataset_summary(trainset, view_data_batch_size, prompt_model, log_file=None, verbose=False):
    if verbose:
        print("\nBootstrapping dataset summary (this will be used to generate instructions)...")
    
    upper_lim = min(len(trainset), view_data_batch_size)
    prompt_model = prompt_model if prompt_model else settings.lm
    with settings.context(lm=prompt_model):
        observation = dataset_descriptor(
                        inputs={"examples":order_input_keys_in_string(repr(trainset[0:upper_lim]))})
