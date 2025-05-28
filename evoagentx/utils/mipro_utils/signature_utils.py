import dspy
from typing import Optional, Dict
from optimizers.engine.registry import ParamRegistry

def build_signature_class(
    registry: ParamRegistry,
    input_descs: Optional[Dict[str, str]] = None,
    output_name: str = "score",
    output_desc: str = "Final evaluation score of the agent output",
    output_type: type = float
):
    """
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
        class_namespace[name] = dspy.InputField(
            desc=input_descs.get(name, f"Tunable parameter: {name}")
        )

    # Add output field with specified configuration
    annotations[output_name] = output_type
    class_namespace[output_name] = dspy.OutputField(desc=output_desc)
    class_namespace['__annotations__'] = annotations

    # Create the signature class dynamically
    class PromptTuningSignature(dspy.Signature):
        __doc__ = class_namespace['__doc__']
        __annotations__ = annotations
        for k, v in class_namespace.items():
            if k not in ('__doc__', '__annotations__'):
                locals()[k] = v

    return PromptTuningSignature
