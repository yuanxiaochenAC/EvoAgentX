import unittest
from unittest.mock import Mock 
from evoagentx.evaluators.evaluator import Evaluator
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.workflow.workflow import WorkFlow

class TestEvaluator(unittest.TestCase):

    def setUp(self):
        # Create mock benchmark
        self.benchmark = Mock(spec=Benchmark)
        self.benchmark.get_test_data.return_value = [
            {"id": "1", "input": "test1"},
            {"id": "2", "input": "test2"}
        ]
        self.benchmark.get_id.side_effect = lambda example: example["id"]
        self.benchmark.get_label.return_value = "expected"
        self.benchmark.evaluate.return_value = {"accuracy": 1.0}

        # Create mock workflow
        self.workflow = Mock(spec=WorkFlow)
        self.workflow.execute.return_value = "prediction"

        # Create evaluator instance
        self.evaluator = Evaluator(num_workers=1)

    def test_single_thread_evaluation(self):
        # Test evaluation with single thread
        results = self.evaluator.evaluate(
            workflow=self.workflow,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Verify results
        self.assertEqual(results, {"accuracy": 1.0})
        self.assertEqual(len(self.evaluator.get_all_evaluation_records()), 2)
        self.assertEqual(self.workflow.execute.call_count, 2)

    def test_multi_thread_evaluation(self):
        # Test evaluation with multiple threads
        evaluator = Evaluator(num_workers=2, verbose=True)
        results = evaluator.evaluate(
            workflow=self.workflow,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Verify results
        self.assertEqual(results, {"accuracy": 1.0})
        self.assertEqual(len(evaluator.get_all_evaluation_records()), 2)
        self.assertEqual(self.workflow.execute.call_count, 2)

    def test_evaluation_with_custom_collate(self):
        # Test evaluation with custom collate function
        def collate_func(x):
            return {"processed_" + k: v for k, v in x.items()}

        evaluator = Evaluator(num_workers=1, collate_func=collate_func)
        evaluator.evaluate(
            workflow=self.workflow,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Verify collate function was applied
        call_args = self.workflow.execute.call_args_list[0][1]["inputs"]
        self.assertTrue(all(k.startswith("processed_") for k in call_args.keys()))
    
    def test_evaluation_with_dict_output_postprocess(self):
        # Mock workflow that returns a dictionary
        workflow = Mock(spec=WorkFlow)
        workflow.execute.return_value = {"result": "prediction"}

        # Define a custom output postprocess function for dictionary output
        def postprocess_func(x):
            return x["result"].upper()

        # Create evaluator with custom postprocess function
        evaluator = Evaluator(num_workers=1, output_postprocess_func=postprocess_func)
        
        # Run evaluation
        evaluator.evaluate(
            workflow=workflow,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Verify that postprocess function was applied
        records = evaluator.get_all_evaluation_records()
        for record in records.values():
            self.assertEqual(record["prediction"], "PREDICTION")
    
    def test_get_example_evaluation_records(self):
        # Run evaluation first
        self.evaluator.evaluate(
            workflow=self.workflow,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Test getting records for specific example
        example = {"id": "1", "input": "test1"}
        record = self.evaluator.get_example_evaluation_records(self.benchmark, example)
        
        self.assertIsNotNone(record)
        self.assertEqual(record["prediction"], "prediction")
        self.assertEqual(record["label"], "expected")
        self.assertEqual(record["metrics"], {"accuracy": 1.0})

    def test_invalid_eval_mode(self):
        # Test that invalid eval_mode raises assertion error
        with self.assertRaises(AssertionError):
            self.evaluator.evaluate(
                workflow=self.workflow,
                benchmark=self.benchmark,
                eval_mode="invalid"
            )

    def test_empty_data_evaluation(self):
        # Test evaluation with empty data
        self.benchmark.get_test_data.return_value = []
        results = self.evaluator.evaluate(
            workflow=self.workflow,
            benchmark=self.benchmark,
            eval_mode="test"
        )
        
        self.assertEqual(results, {})
        self.assertEqual(len(self.evaluator.get_all_evaluation_records()), 0)

if __name__ == '__main__':
    unittest.main()