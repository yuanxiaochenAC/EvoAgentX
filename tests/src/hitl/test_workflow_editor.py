import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from evoagentx.hitl.workflow_editor import WorkFlowEditor, WorkFlowEditorReturn
from evoagentx.models import OpenAILLM, OpenAILLMConfig


class TestWorkFlowEditor(unittest.IsolatedAsyncioTestCase):
    """Test the WorkFlowEditor class"""
    
    def setUp(self):
        """Test preparation"""
        # create a temporary directory as the save directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_workflow_file = os.path.join(os.path.dirname(__file__), 
                                               '..', '..', '..', 
                                               'examples', 'output', 'tetris_game', 
                                               'workflow_demo_4o_mini.json')
        
        # create a mock LLM configuration
        self.mock_llm_config = OpenAILLMConfig(
            model="gpt-4o-mini", 
            openai_key="test_key", 
            stream=False, 
            output_response=True
        )
        
        # create a mock LLM instance
        self.mock_llm = MagicMock(spec=OpenAILLM)
        
        # prepare the test instruction
        self.test_instruction = "delete the last node which is not useful in our case"
        
        # create the expected optimized JSON structure (delete the last node)
        with open(self.test_workflow_file, 'r', encoding='utf-8') as f:
            original_workflow = json.load(f)
        
        # delete the last node and the related edges
        self.expected_optimized_workflow = original_workflow.copy()
        if self.expected_optimized_workflow['nodes']:
            # delete the last node
            last_node = self.expected_optimized_workflow['nodes'][-1]
            self.expected_optimized_workflow['nodes'] = self.expected_optimized_workflow['nodes'][:-1]
            
            # delete the related edges
            self.expected_optimized_workflow['edges'] = [
                edge for edge in self.expected_optimized_workflow['edges'] 
                if edge['target'] != last_node['name'] and edge['source'] != last_node['name']
            ]

    def tearDown(self):
        """Test cleanup"""
        # delete the temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_workflow_editor_instantiation(self):
        """Test the instantiation of the WorkFlowEditor class"""
        # test using default parameters
        editor = WorkFlowEditor(save_dir=self.temp_dir)
        
        # verify the instantiation is successful
        self.assertIsInstance(editor, WorkFlowEditor)
        self.assertEqual(editor.save_dir, self.temp_dir)
        self.assertEqual(editor.max_retries, 3)
        self.assertIsNotNone(editor.llm)
        
        # test using custom parameters
        editor_custom = WorkFlowEditor(
            save_dir=self.temp_dir,
            llm=self.mock_llm,
            max_retries=5
        )
        
        self.assertIsInstance(editor_custom, WorkFlowEditor)
        self.assertEqual(editor_custom.save_dir, self.temp_dir)
        self.assertEqual(editor_custom.max_retries, 5)
        self.assertEqual(editor_custom.llm, self.mock_llm)

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_edit_workflow_without_new_file_path(self, mock_workflow_graph, mock_workflow):
        """Test the edit_workflow method (without providing the new_file_path parameter)"""
        # set the mock objects
        mock_workflow_graph.from_dict.return_value = MagicMock()
        mock_workflow.return_value = MagicMock()
        
        # create the editor instance
        editor = WorkFlowEditor(save_dir=self.temp_dir, llm=self.mock_llm)
        
        # mock the LLM to return the optimized JSON
        self.mock_llm.single_generate_async.return_value = json.dumps(self.expected_optimized_workflow)
        
        # call the edit_workflow method (without providing the new_file_path parameter)
        result = await editor.edit_workflow(
            file_path=self.test_workflow_file,
            instruction=self.test_instruction
        )
        
        # verify the result
        self.assertIsInstance(result, WorkFlowEditorReturn)
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.workflow_json)
        self.assertIsNotNone(result.workflow_json_path)
        self.assertIsNone(result.error_message)
        
        # verify the file is created
        self.assertTrue(os.path.exists(result.workflow_json_path))
        
        # verify the file name format (should contain the timestamp)
        self.assertIn("new_json_for__", os.path.basename(result.workflow_json_path))
        self.assertTrue(result.workflow_json_path.endswith('.json'))
        
        # verify the content of the generated JSON file
        with open(result.workflow_json_path, 'r', encoding='utf-8') as f:
            saved_json = json.load(f)
        
        self.assertEqual(saved_json, self.expected_optimized_workflow)
        
        # verify the LLM is called correctly
        self.mock_llm.single_generate_async.assert_called_once()
        
        # clean up the generated file
        if os.path.exists(result.workflow_json_path):
            os.remove(result.workflow_json_path)

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_edit_workflow_with_new_file_path(self, mock_workflow_graph, mock_workflow):
        """Test the edit_workflow method (with the new_file_path parameter)"""
        # set the mock objects
        mock_workflow_graph.from_dict.return_value = MagicMock()
        mock_workflow.return_value = MagicMock()
        
        # create the editor instance
        editor = WorkFlowEditor(save_dir=self.temp_dir, llm=self.mock_llm)
        
        # mock the LLM to return the optimized JSON
        self.mock_llm.single_generate_async.return_value = json.dumps(self.expected_optimized_workflow)
        
        # define the temporary file path
        temp_file_name = "test_optimized_workflow.json"
        temp_file_path = os.path.join(self.temp_dir, temp_file_name)
        
        # call the edit_workflow method (with the new_file_path parameter)
        result = await editor.edit_workflow(
            file_path=self.test_workflow_file,
            instruction=self.test_instruction,
            new_file_path=temp_file_name
        )
        
        # verify the result
        self.assertIsInstance(result, WorkFlowEditorReturn)
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.workflow_json)
        self.assertEqual(result.workflow_json_path, temp_file_path)
        self.assertIsNone(result.error_message)
        
        # verify the file is created
        self.assertTrue(os.path.exists(result.workflow_json_path))
        
        # verify the content of the generated JSON file
        with open(result.workflow_json_path, 'r', encoding='utf-8') as f:
            saved_json = json.load(f)
        
        self.assertEqual(saved_json, self.expected_optimized_workflow)
        
        # verify the LLM is called correctly
        self.mock_llm.single_generate_async.assert_called_once()
        
        # clean up the generated file
        if os.path.exists(result.workflow_json_path):
            os.remove(result.workflow_json_path)

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_edit_workflow_llm_failure(self, mock_workflow_graph, mock_workflow):
        """Test the edit_workflow method when the LLM fails"""
        # create the editor instance
        editor = WorkFlowEditor(save_dir=self.temp_dir, llm=self.mock_llm)
        
        # mock the LLM to throw an exception
        self.mock_llm.single_generate_async.side_effect = Exception("LLM failure")
        
        # call the edit_workflow method
        result = await editor.edit_workflow(
            file_path=self.test_workflow_file,
            instruction=self.test_instruction
        )
        
        # verify the result
        self.assertIsInstance(result, WorkFlowEditorReturn)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.workflow_json)
        self.assertIsNone(result.workflow_json_path)
        self.assertEqual(result.error_message, "LLM optimization failed")

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_edit_workflow_invalid_json_structure(self, mock_workflow_graph, mock_workflow):
        """Test the edit_workflow method when the workflow JSON structure validation fails"""
        # create the editor instance
        editor = WorkFlowEditor(save_dir=self.temp_dir, llm=self.mock_llm)
        
        # mock the LLM to return valid JSON but the structure validation fails
        self.mock_llm.single_generate_async.return_value = json.dumps({"invalid": "structure"})
        mock_workflow_graph.from_dict.side_effect = Exception("Invalid structure")
        
        # call the edit_workflow method
        result = await editor.edit_workflow(
            file_path=self.test_workflow_file,
            instruction=self.test_instruction
        )
        
        # verify the result
        self.assertIsInstance(result, WorkFlowEditorReturn)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.workflow_json)
        self.assertIsNone(result.workflow_json_path)
        self.assertEqual(result.error_message, "Workflow json structure check failed")

    async def test_edit_workflow_invalid_file_path(self):
        """Test the edit_workflow method when providing an invalid file path"""
        # create the editor instance
        editor = WorkFlowEditor(save_dir=self.temp_dir, llm=self.mock_llm)
        
        # test the invalid new_file_path
        invalid_path = "/non_existent_directory/test.json"
        
        with self.assertRaises(FileNotFoundError):
            await editor.edit_workflow(
                file_path=self.test_workflow_file,
                instruction=self.test_instruction,
                new_file_path=invalid_path
            )


class TestWorkFlowEditorIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration test class"""
    
    def setUp(self):
        """Test preparation"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_workflow_file = os.path.join(os.path.dirname(__file__), 
                                               '..', '..', '..', 
                                               'examples', 'output', 'tetris_game', 
                                               'workflow_demo_4o_mini.json')
        self.test_instruction = "delete the last node which is not useful in our case"
        
        # create a mock LLM instance
        self.mock_llm = MagicMock(spec=OpenAILLM)
        
        # prepare the mock optimized JSON
        with open(self.test_workflow_file, 'r', encoding='utf-8') as f:
            original_workflow = json.load(f)
        
        self.optimized_workflow = original_workflow.copy()
        if self.optimized_workflow['nodes']:
            last_node = self.optimized_workflow['nodes'][-1]
            self.optimized_workflow['nodes'] = self.optimized_workflow['nodes'][:-1]
            self.optimized_workflow['edges'] = [
                edge for edge in self.optimized_workflow['edges'] 
                if edge['target'] != last_node['name'] and edge['source'] != last_node['name']
            ]

    def tearDown(self):
        """Test cleanup"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_full_workflow_editing_process(self, mock_workflow_graph, mock_workflow):
        """Test the full workflow editing process"""
        # set the mock objects
        mock_workflow_graph.from_dict.return_value = MagicMock()
        mock_workflow.return_value = MagicMock()
        
        # create the editor instance
        editor = WorkFlowEditor(save_dir=self.temp_dir, llm=self.mock_llm)
        
        # mock the LLM to return the optimized JSON
        self.mock_llm.single_generate_async.return_value = json.dumps(self.optimized_workflow)
        
        # test case 1: without providing the new_file_path parameter
        result1 = await editor.edit_workflow(
            file_path=self.test_workflow_file,
            instruction=self.test_instruction
        )
        
        self.assertEqual(result1.status, "success")
        self.assertTrue(os.path.exists(result1.workflow_json_path))
        
        # test case 2: with the new_file_path parameter
        temp_file_name = "integration_test_output.json"
        result2 = await editor.edit_workflow(
            file_path=self.test_workflow_file,
            instruction=self.test_instruction,
            new_file_path=temp_file_name
        )
        
        self.assertEqual(result2.status, "success")
        self.assertTrue(os.path.exists(result2.workflow_json_path))
        
        # clean up the generated files
        if os.path.exists(result1.workflow_json_path):
            os.remove(result1.workflow_json_path)
        if os.path.exists(result2.workflow_json_path):
            os.remove(result2.workflow_json_path)


if __name__ == '__main__':
    # run the tests
    unittest.main() 