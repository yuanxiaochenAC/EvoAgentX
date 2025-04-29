import json
from pydantic import Field
from typing import Dict, Any, List

from ..core.logging import logger
from ..core.module import BaseModule
from ..core.registry import MODEL_REGISTRY, MODULE_REGISTRY
from ..models.model_configs import LLMConfig
from .operators import Operator, AnswerGenerate, QAScEnsemble 


class ActionGraph(BaseModule):
    """
    Base class for action graphs that define workflows with operators.
    
    An action graph represents a workflow that can be executed with a language model.
    It contains a collection of operators that perform specific tasks within the workflow.
    """

    name: str = Field(description="The name of the ActionGraph.")
    description: str = Field(description="The description of the ActionGraph.")
    llm_config: LLMConfig = Field(description="The config of LLM used to execute the ActionGraph.")

    def init_module(self):
        """
        Initialize the module by setting up the language model.
        
        This method is called during module initialization to create the language model
        instance based on the provided configuration.
        """
        if self.llm_config:
            llm_cls = MODEL_REGISTRY.get_model(self.llm_config.llm_type)
            self._llm = llm_cls(config=self.llm_config)
    
    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """
        Allow the ActionGraph to be called directly.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            dict: The result of the execute method.
        """
        return self.execute(*args, **kwargs)
    
    def execute(self, *args, **kwargs) -> dict:
        """
        Execute the action graph workflow.
        
        This method should be implemented by subclasses to define the specific
        execution logic for the workflow.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            dict: The result of the workflow execution.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError(f"The execute function for {type(self).__name__} is not implemented!")
    
    def get_graph_info(self, **kwargs) -> dict:
        """
        Get the information of the action graph, including all operators from the instance.
        
        This method collects information about the graph, including its class name,
        name, description, LLM configuration, and all operators.
        
        Args:
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            dict: A dictionary containing the graph information.
        """
        operators = {}
        # the extra fields are the fields that are not defined in the Pydantic model 
        for extra_name, extra_value in self.__pydantic_extra__.items():
            if isinstance(extra_value, Operator):
                operators[extra_name] = extra_value

        config = {
            "class_name": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "llm_config": self.llm_config.to_dict(exclude_none=True) \
                if self.llm_config is not None else \
                    self._llm.config.to_dict(exclude_none=True),
            "operators": {
                operator_name: {
                    "class_name": operator.__class__.__name__,
                    "name": operator.name,
                    "description": operator.description,
                    "interface": operator.interface,
                    "prompt": operator.prompt
                }
                for operator_name, operator in operators.items()
            }
        }
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "ActionGraph":
        """
        Create an ActionGraph from a dictionary.
        
        This class method creates an ActionGraph instance from a dictionary containing
        the graph configuration and operator information.
        
        Args:
            data: A dictionary containing the graph configuration.
            **kwargs: Additional keyword arguments.
            
        Returns:
            ActionGraph: An instance of ActionGraph or its subclass.
        """
        class_name = data.get("class_name", None)
        if class_name:
            cls = MODULE_REGISTRY.get_module(class_name)
        operators_info = data.pop("operators", None)
        module = cls._create_instance(data)
        if operators_info:
            for extra_name, extra_value in module.__pydantic_extra__.items():
                if isinstance(extra_value, Operator) and extra_name in operators_info:
                    extra_value.set_operator(operators_info[extra_name])
        return module
    
    def save_module(self, path: str, ignore: List[str] = [], **kwargs):
        """
        Save the workflow graph to a module file.
        
        This method saves the graph configuration to a JSON file.
        
        Args:
            path: The file path where the module will be saved.
            ignore: A list of keys to ignore when saving the module.
            **kwargs: Additional keyword arguments.
            
        Returns:
            str: The path where the module was saved.
        """
        logger.info("Saving {} to {}", self.__class__.__name__, path)
        config = self.get_graph_info()
        for ignore_key in ignore:
            config.pop(ignore_key, None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        return path


class QAActionGraph(ActionGraph):
    """
    A question-answering action graph that uses self-consistency for predictions.
    
    This graph implements a workflow for answering questions using a language model
    with self-consistency ensemble to improve answer quality.
    """

    def __init__(self, llm_config: LLMConfig, **kwargs):
        """
        Initialize the QA action graph.
        
        Args:
            llm_config: Configuration for the language model.
            **kwargs: Additional keyword arguments.
        """
        name = kwargs.pop("name") if "name" in kwargs else "Simple QA Workflow"
        description = kwargs.pop("description") if "description" in kwargs else \
            "This is a simple QA workflow that use self-consistency to make predictions."
        super().__init__(name=name, description=description, llm_config=llm_config, **kwargs)
        self.answer_generate = AnswerGenerate(self._llm)
        self.sc_ensemble = QAScEnsemble(self._llm)
        
    def execute(self, problem: str) -> dict:
        """
        Execute the QA workflow to answer a question.
        
        This method generates multiple answers to the same question and then
        uses self-consistency ensemble to select the best answer.
        
        Args:
            problem: The question to be answered.
            
        Returns:
            dict: A dictionary containing the best answer.
        """
        solutions = [] 
        for _ in range(3):
            response = self.answer_generate(input=problem)
            answer = response["answer"]
            solutions.append(answer)
        ensemble_result = self.sc_ensemble(solutions=solutions)
        best_answer = ensemble_result["response"]
        return {"answer": best_answer}
    