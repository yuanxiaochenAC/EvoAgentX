
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


dataset_descriptor_with_prior = CustomizeAgent(
    name="DatasetDescriptorWithPrior",
    description="An agent that describes a dataset given prior observations",
    prompt="""
    Given several examples from a dataset please write observations about trends that hold for most or all of the samples.
    I will also provide you with a few observations I have already made. Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE'
    Some areas you may consider in your observations: topics, content, syntax, conciceness, etc.
    It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative
    """,
    inputs = [
        {"name": "examples", "type": "list", "desc": "Sample data points from the dataset"},
        {"name": "prior_observations", "type": "str", "desc": "Some prior observations I made about the data"}
    ],
    outputs = [
        {"name": "observations", "type": "str", "desc": "Somethings that holds true for most or all of the data you observed or COMPLETE if you have nothing to add"}
    ],
    llm_config=settings.lm,
    parse_mode=str,
)


observation_summarizer = CustomizeAgent(
    name="ObservationSummarizer",
    description="An agent that summarizes dataset observations",
    prompt="""
    Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.
    """,
    inputs = [
        {"name": "observations", "type": "str", "desc": "Observations I have made about my dataset"}
    ],
    outputs = [
        {"name": "summary", "type": "str", "desc": "Two to Three sentence summary of only the most significant highlights of my observations"}
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
                        inputs={"examples":order_input_keys_in_string(repr(trainset[0:upper_lim]))},
                        llm_config=settings.lm.copy(temperature=1.0, n=1))

        observations = observation.content.observations
        
    if log_file:
        log_file.write("PRODUCING DATASET SUMMARY\n")
    
    skips = 0
    try:
        max_calls = 10
        calls = 0
        for b in range(view_data_batch_size, len(trainset), view_data_batch_size):
            calls+=1
            if calls >= max_calls:
                break
            if verbose:
                print(f"b: {b}")
            upper_lim = min(len(trainset), b+view_data_batch_size)
            with settings.context(lm=prompt_model):
                output = dataset_descriptor_with_prior(
                    inputs={ "prior_observations": observations, "examples": order_input_keys_in_string(repr(trainset[b:upper_lim]))},
                    llm_config=settings.lm.copy(temperature=1.0, n=1))

                output_observations = output.content.observations
                if len(output_observations) >= 8 and output_observations[:8].upper() == "COMPLETE":
                    skips += 1
                    if skips >= 5:
                        break
                    continue
                observations += output_observations

                if log_file:
                    log_file.write(f"Observations: {observations}\n")

    except Exception as e:
        if verbose:
            print(f"e {e}. using observations from past round for a summary.")
    
    if prompt_model:
        with settings.context(lm=prompt_model):
            summary = observation_summarizer(
                inputs={"observations": observations},
                llm_config=settings.lm.copy(temperature=1.0, n=1)
            )
    else:
        summary = observation_summarizer(
            inputs={"observations": observations},
            llm_config=settings.lm.copy(temperature=1.0, n=1)
        )
            
    if verbose:
        print(f"summary: {summary}")
    if log_file:
        log_file.write(f"summary: {summary}\n")
    
    if verbose:
        print(f"\nGenerated summary: {strip_prefix(summary.content.summary)}\n")

    return strip_prefix(summary.content.summary)
