from optimizers.engine.registry import ParamRegistry

from dspy import Signature, InputField, OutputField
from typing import Optional, Dict, Any, Type, Union
from ...prompts.template import PromptTemplate


def signature_from_registry(
    registry: ParamRegistry,
    output_field_name: str = "output",
    output_field_desc: str = "The final output.",
    output_field_type: Type = str,
    custom_output_fields: Optional[Dict[str, OutputField]] = None,
) -> Dict[str, Type[Signature]]:
    """
    Creates Signature classes for each field in the registry.

    This function dynamically generates DSPy Signature classes based on the contents
    of a parameter registry. It handles two types of registry values:
    1. String prompts: Creates a single input field using the registry name
    2. PromptTemplates: Creates multiple input fields based on non-empty template fields

    Parameters
    ----------
    registry : ParamRegistry
        The registry containing fields to convert into signatures
    output_field_name : str, default="output"
        Default name for the output field in generated signatures
    output_field_desc : str, default="The final output."
        Default description for the output field
    output_field_type : Type, default=str
        Type annotation for the output field. This type will be used in the 
        signature's __annotations__ for type checking and IDE support.
    custom_output_fields : Optional[Dict[str, OutputField]], default=None
        Optional mapping of field names to custom output field configurations.
        If provided, these will override the default output field settings.

    Returns
    -------
    Dict[str, Type[Signature]]
        A dictionary mapping field names to their corresponding Signature subclasses

    Examples
    --------
    >>> registry = ParamRegistry()
    >>> registry.register("task1", "What is {topic}?")
    >>> registry.register("task2", PromptTemplate(system="You are helpful.", user="{query}"))
    >>> signatures = signature_from_registry(registry, output_field_type=bool)
    >>> # signatures["task1"] will have input field "task1" and output field "output" of type bool
    >>> # signatures["task2"] will have input fields "system", "user" and output field "output" of type bool
    """
    sigs = {}
    for name in registry.names():
        value: Union[str, PromptTemplate] = registry.get(name)

        # Create input fields based on registry value type
        input_fields = []
        if isinstance(value, str):
            # For string prompts, use registry name as the input field
            input_fields = [InputField(name, desc=f"The prompt for task `{name}`.")]
        elif isinstance(value, PromptTemplate):
            # For PromptTemplates, create fields for each non-empty template field
            for field_name in value.get_field_names():
                val = getattr(value, field_name)
                if val is not None:
                    input_fields.append(InputField(
                        field_name, 
                        desc=f"Field `{field_name}` from PromptTemplate `{name}`."
                    ))
        else:
            # Skip unsupported value types
            continue

        # Configure output field - use custom if provided, otherwise use defaults
        output_field = (
            custom_output_fields[name] 
            if custom_output_fields and name in custom_output_fields 
            else OutputField(
                name=output_field_name,
                desc=output_field_desc,
                annotation=output_field_type  # Add type annotation to output field
            )
        )

        # Validate: ensure output field name doesn't conflict with input fields
        if output_field.name in [f.name for f in input_fields]:
            raise ValueError(
                f"Output field name `{output_field.name}` conflicts with input fields in `{name}`."
            )

        # Construct the Signature class
        sig_dict = {f.name: f for f in input_fields}  # Convert input fields to dictionary
        sig_dict[output_field.name] = output_field    # Add output field

        # Create dynamic Signature subclass with proper type annotations
        sig_class = type(f"{name}Signature", (Signature,), sig_dict)
        sigs[name] = sig_class

    return sigs


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