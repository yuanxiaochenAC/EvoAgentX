from ...optimizers.engine.registry import ParamRegistry

from dspy import Signature, InputField, OutputField
from typing import Optional, Dict, Type, Union
from ...prompts.template import PromptTemplate


def signature_from_registry(
    registry: ParamRegistry,
    output_field_name: str = "output",
    output_field_desc: str = "The final output.",
    output_field_type: Type = str,
    custom_output: Optional[Dict[str, str]] = None,
) -> Dict[str, Type[Signature]]:
    """
    Creates Signature classes for each field in the registry.

    This function dynamically generates DSPy Signature classes based on the contents
    of a parameter registry. It handles two types of registry values:
    1. String prompts: Creates a signature with a single input field and an output field
    2. PromptTemplates: Creates a signature with multiple input fields (one for each template field) and an output field

    Parameters
    ----------
    registry : ParamRegistry
        The registry containing fields to convert into signatures
    output_field_name : str, default="output"
        Name for the output field in generated signatures
    output_field_desc : str, default="The final output."
        Default description for the output field
    output_field_type : Type, default=str
        Type annotation for the output field
    custom_output : Optional[Dict[str, str]], default=None
        Optional mapping of field names to custom output field descriptions.
        If provided, these will override the default output field description.

    Returns
    -------
    Dict[str, Type[Signature]]
        A dictionary mapping field names to their corresponding Signature subclasses

    Examples
    --------
    >>> registry = ParamRegistry()
    >>> registry.register("task1", "What is {topic}?")
    >>> registry.register("task2", PromptTemplate(system="You are helpful.", user="{query}"))
    >>> signatures = signature_from_registry(registry)
    >>> # signatures["task1"] will have:
    >>> #   - input field "task1" with description "The prompt for task `task1`"
    >>> #   - output field "output" with description "The final output"
    >>> # signatures["task2"] will have:
    >>> #   - input fields "system" and "user" with descriptions from the template
    >>> #   - output field "output" with description "The final output"
    """
    signature_dict = {}
    for name in registry.names():
        value: Union[str, PromptTemplate] = registry.get(name)
        sig_dict = {}

        if isinstance(value, str):
            # For string prompts, create a simple signature with one input field
            sig_dict[name] = InputField(desc=f"The prompt for task `{name}`.")
            
        elif isinstance(value, PromptTemplate):
            # For PromptTemplates, create input fields for each template field
            for field_name in value.get_field_names():
                val = getattr(value, field_name)
                if val is not None:
                    sig_dict[field_name] = InputField(desc=f"Field `{field_name}` from PromptTemplate `{name}`.")
        else:
            # Skip unsupported value types
            continue

        # Add output field with custom description if provided
        output_desc = custom_output[name] if custom_output and name in custom_output else output_field_desc
        sig_dict[output_field_name] = OutputField(
            desc=output_desc,
            annotation=output_field_type
        )

        # Create the Signature class
        sig_class = type(f"{name}Signature", (Signature,), sig_dict)
        signature_dict[name] = sig_class

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