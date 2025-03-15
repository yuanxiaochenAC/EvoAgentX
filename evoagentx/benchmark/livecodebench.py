import os 
import regex
from typing import Union, Dict, Any, List
from ..core.logging import logger
from .benchmark import CodingBenchmark 
from ..utils.utils import download_file
from ..core.module_utils import load_json
from ..core.module_utils import extract_code_blocks
from .lcb_utils.code_generation import (
    CodeGenerationProblem, 
    load_code_generation_dataset
)
from .lcb_utils.evaluation import codegen_metrics 


VALID_SCENARIO = ["code_generation", "code_repair", "code_execution", "test_output_prediction"]

class LiveCodeBench(CodingBenchmark):

    """
    scenario: "code_generation"
    CodeGenerationProblem(
        question_title: str
        question_content: str
        platform: Platform
        question_id: str
        contest_id: str
        contest_date: datetime
        starter_code: str
        difficulty: Difficulty
        public_test_cases: list[Test]
        private_test_cases: list[Test]
        metadata: dict
    )
    """

    def __init__(
        self, 
        path: str = None, 
        mode: str = "all", 
        timeout: int = 60, 
        k: Union[int, list] = 1, 
        num_process: int = 6, 
        scenario: str = "code_generation", 
        version: str = "release_latest", 
        start_date: str = None, 
        end_date: str = None, 
        **kwargs
    ):
        path = os.path.expanduser(path or "~/.evoagentx/data/livecodebench")
        self.k = k 
        self.version = version
        self.num_process = num_process
        self.start_date = start_date
        self.end_date = end_date
        assert scenario in VALID_SCENARIO, f"Invalid scenario: {scenario}. Available choices: {VALID_SCENARIO}." 
        self.scenario = scenario 
        super().__init__(name=type(self).__name__, path=path, mode=mode, timeout=timeout, **kwargs)
    
    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = None 
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_test_data()
    
    def _load_test_data(self):

        if self.scenario == "code_generation":
            logger.info(f"Loading code generation dataset from {self.path} with version {self.version}.")
            data: List[CodeGenerationProblem] = load_code_generation_dataset(
                release_version=self.version, 
                cache_dir=self.path, 
                start_date=self.start_date, 
                end_date=self.end_date
            )
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}. Available choices: {VALID_SCENARIO}.")

        return data 
    
    def _get_id(self, example: Union[CodeGenerationProblem]) -> str:
        if isinstance(example, CodeGenerationProblem):
            return example.question_id 
        else:
            raise ValueError(f"Invalid example type: {type(example)}. Expected CodeGenerationProblem.")
    
    def _get_label(self, example: Union[CodeGenerationProblem]) -> dict:
        if isinstance(example, CodeGenerationProblem):
            return example.get_evaluation_sample()
        else:
            raise ValueError(f"Invalid example type: {type(example)}. Expected CodeGenerationProblem.")
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Evaluate the solution code.

        Args:
            prediction (str | List[str]): The solution code(s).
            label (dict | List[dict]): The test cases and expected outputs. 

        Returns:
            dict: The evaluation metrics (pass@k).
        """
        prediction, label = self._check_evaluation_inputs(prediction, label)
        solutions: List[str] = [extract_code_blocks(pred)[0] for pred in prediction]

        k_list = [self.k] if isinstance(self.k, int) else self.k
        if self.scenario == "code_generation":
            metrics, results, metadatas = codegen_metrics(
                samples_list=label, # label is already a list 
                generations_list=[solutions], # for a single example. 
                k_list=k_list, 
                num_process_evaluate=self.num_process,
                timeout=self.timeout
            )
            pass_at_k = {f"pass@{k}": float(metrics[f"pass@{k}"]) for k in k_list}
            return pass_at_k
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}. Available choices: {VALID_SCENARIO}.")
