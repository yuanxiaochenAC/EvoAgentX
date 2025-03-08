import os 
from typing import Dict, Any, List
from ..core.logging import logger
from .benchmark import Benchmark 
from ..utils.utils import download_file
from ..core.module_utils import load_json


def download_raw_mbpp_data(name: str, save_folder: str):
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
    logger.info(f"Downloading MBPP data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


class MBPP(Benchmark):

    """
    {
        "text" (str): "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
        "code" (str): "def xxx()", 
        "task_id" (int): 1, 
        "test_setup_code" (str): "", 
        "test_list" (List[str]): ['assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8', 'assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8'],
        "challenge_test_list" (List[str]): []
    }
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/mbpp")
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)
    
    def _load_data(self):
        mbpp_file_name = "mbpp.jsonl"
        file_path = os.path.join(self.path, mbpp_file_name)
        if not os.path.exists(file_path):
            download_raw_mbpp_data(name=mbpp_file_name, save_folder=self.path)
        # loading data from file 
        logger.info(f"loading MBPP data from {file_path} ...")
        data = load_json(path=file_path, type="jsonl")

        # load train, dev, test data: https://github.com/google-research/google-research/tree/master/mbpp
        # MBPP uses IDs 1-10 for few-shot prompting, 11-510 for test, 511-600 for validation, and 601-974 for training
        id2data = {example["task_id"]: example for example in data}
        if self.mode == "train" or self.mode == "all":
            self._train_data = self.get_data_based_on_task_ids(id2data=id2data, task_ids=range(601, 975))
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self.get_data_based_on_task_ids(id2data=id2data, task_ids=range(511, 601))
        if self.mode == "test" or self.mode == "all":
            self._test_data = self.get_data_based_on_task_ids(id2data=id2data, task_ids=range(11, 511))

    def get_data_based_on_task_ids(self, id2data: Dict[int, dict], task_ids: List[int]):
        examples = [id2data[tid] for tid in task_ids]
        return examples
    
    def _get_id(self, example: Any) -> Any:
        return example["task_id"]

    def _get_label(self, example: Any) -> Any:
        return example["test_list"]
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        pass
    
    