


from dspy import Signature, InputField, OutputField
from typing import Optional, Dict, Type, Union
from ...prompts.template import PromptTemplate
from ...optimizers.engine.registry import ParamRegistry
from ...utils.mipro_utils.register_utils import MiproRegister
from dspy.signatures.signature import make_signature
import keyword
import re
import warnings

def is_valid_identifier(key: str) -> bool:
    return key.isidentifier() and not keyword.iskeyword(key)

def check_input_placeholders(instruction: str, input_names: list[str], key: str):
    placeholders = set(re.findall(r"\{(\w+)\}", instruction))
    input_names_set = set(input_names or [])

    missing = placeholders - input_names_set
    if missing:
        warnings.warn(
            f"[{key}] Missing input_names for placeholders in instruction: {missing}"
        )

def signature_from_registry(
    registry: MiproRegister,
) -> Dict[str, Type[Signature]]:
    
    signature_dict = {}
    for key in registry.names():
        registered_element: Union[str, PromptTemplate] = registry.get(key)
        input_names = registry.get_input_names(key)
        output_names = registry.get_output_names(key)
        sig = {}

        # sig_dict[key] = (str, InputField(desc=f"The Input for prompt `{key}`."))

        if isinstance(registered_element, str):
            # For string prompts, create a simple signature with one input field
            instructions = registered_element
            
        elif isinstance(registered_element, PromptTemplate):
            instructions = registered_element.instruction
            # for field_name in registered_element.get_field_names():
        
        check_input_placeholders(instructions, input_names, key)

        for name in input_names:
            input_desc = registry.get_input_desc(key, name)
            if input_desc:
                sig[name] = (str, InputField(desc=input_desc))
            else:
                sig[name] = (str, InputField(desc=f"The Input for prompt `{key}`."))

        for name in output_names:
            output_desc = registry.get_output_desc(key, name)
            if output_desc:
                sig[name] = (str, OutputField(desc=output_desc))
            else:
                sig[name] = (str, OutputField(desc=f"The Output for prompt `{key}`."))

        if is_valid_identifier(key):
            signature_name = f"{key}Signature"
        else:
            # if the key is not a valid identifier, we need to add an underscore
            # 打印warning
            print(f"Warning: The key `{key}` is not a valid identifier, so we will add an underscore to it.")
            signature_name = f"DefaultSignature_{len(signature_dict)}"

        signature_class = make_signature(signature=sig, instructions=instructions, signature_name=signature_name)

        signature_dict[key] = signature_class


    return signature_dict


# Unused Function
def build_signature_class(
    registry: ParamRegistry,
    input_descs: Optional[Dict[str, str]] = None,
    output_name: str = "score",
    output_desc: str = "Final evaluation score of the agent output",
    output_type: type = float
):
    """
    unused function
    Dynamically builds a DSPy Signature class based on a parameter registry.
    
    This function creates a new DSPy Signature class that defines input and output fields
    based on the parameters in the registry. Each parameter becomes an input field in the
    signature, and an additional output field is added for the evaluation score.
    
    Parameters
    ----------
    registry : ParamRegistry
        Registry containing the tunable parameters that will become input fields
    input_descs : Optional[Dict[str, str]], default=None
        Optional descriptions for input parameters. Keys are parameter names,
        values are their descriptions. If not provided for a parameter,
        a default description will be generated.
    output_name : str, default="score"
        Name of the output field in the signature
    output_desc : str, default="Final evaluation score of the agent output"
        Description of the output field
    output_type : type, default=float
        Type annotation for the output field
        
    Returns
    -------
    type
        A new DSPy Signature subclass with dynamically defined input and output fields
        
    Examples
    --------
    >>> registry = ParamRegistry()
    >>> registry.register("temperature", 0.7)
    >>> signature = build_signature_class(
    ...     registry,
    ...     input_descs={"temperature": "Sampling temperature"}
    ... )
    """
    # Initialize empty descriptions dictionary if none provided
    input_descs = input_descs or {}
    
    # Get all parameter names from registry
    fields = registry.names()

    # Prepare class attributes and type annotations
    annotations = {}
    class_namespace = {"__doc__": "Auto-generated signature class."}

    # Create input fields for each parameter in registry
    for name in fields:
        annotations[name] = str  # All inputs are treated as strings
        class_namespace[name] = InputField(
            desc=input_descs.get(name, f"Tunable parameter: {name}")
        )

    # Add output field with specified configuration
    annotations[output_name] = output_type
    class_namespace[output_name] = OutputField(desc=output_desc)
    class_namespace['__annotations__'] = annotations

    # Create the signature class dynamically
    class PromptTuningSignature(Signature):
        __doc__ = class_namespace['__doc__']
        __annotations__ = annotations
        for k, v in class_namespace.items():
            if k not in ('__doc__', '__annotations__'):
                locals()[k] = v

    return PromptTuningSignature