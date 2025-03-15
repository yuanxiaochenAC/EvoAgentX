import unittest
from unittest.mock import patch
import os
import shutil
import tempfile
from datasets import load_from_disk 
from evoagentx.benchmark.livecodebench import LiveCodeBench
from evoagentx.benchmark.lcb_utils.code_generation import CodeGenerationProblem
from tests.src.benchmark.lcb_solutions import codegen_solution, codegen_solution2, codegen_solution3

class TestLiveCodeBench(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.codegen_samples = load_from_disk("tests/data/benchmark/lcb_codegen_samples")
        self.codegen_solutions = [codegen_solution, codegen_solution2, codegen_solution3]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch("evoagentx.benchmark.livecodebench.load_code_generation_dataset")
    def test_code_generation(self, mock_load_dataset):
        mock_load_dataset.return_value = [CodeGenerationProblem(**p) for p in self.codegen_samples]

        benchmark = LiveCodeBench(scenario="code_generation", version="release_v1")
        test_data = benchmark.get_test_data()

        self.assertEqual(len(test_data), len(self.codegen_samples))
        self.assertEqual(mock_load_dataset.call_count, 1)

        for example, solution in zip(test_data, self.codegen_solutions):
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(solution, label) 
            self.assertEqual(metrics, {"pass@1": 1.0})
        