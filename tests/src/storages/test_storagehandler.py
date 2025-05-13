import unittest
import json
from datetime import datetime
from unittest.mock import patch

from evoagentx.storages.storages_config import StoreConfig, DBConfig, VectorStoreConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.schema import TableType


class TestStorageHandler(unittest.TestCase):
    """
    Test suite for StorageHandler's database operations on Workflow, Agent, and History.
    Uses an in-memory SQLite database for isolated testing.
    """
    def setUp(self):
        """
        Set up the test environment by initializing StorageHandler with an in-memory SQLite database.
        """
        # Mock configuration
        db_config = DBConfig(db_name="sqlite", path=":memory:")
        store_config = StoreConfig(dbConfig=db_config)
        
        self.storage = StorageHandler(storageConfig=store_config)
        
        # Sample data for testing
        self.agent_data = {
            "name": "test_agent",
            "content": {"role": "assistant", "settings": {"active": True}},
            "date": "2025-05-13"
        }
        self.workflow_data = {
            "name": "test_workflow",
            "content": {"steps": ["step1", "step2"], "config": {"timeout": 30}},
            "date": "2025-05-13"
        }
        self.history_data = {
            "memory_id": "mem_001",
            "old_memory": "Initial content",
            "new_memory": "Updated content",
            "event": "update",
            "created_at": "2025-05-13T09:00:00",
            "updated_at": "2025-05-13T09:30:00"
        }

    def test_save_and_load_agent(self):
        """
        Test saving and loading an agent, verifying data integrity and JSON parsing.
        """
        # Save agent
        self.storage.save_agent(self.agent_data)
        self.storage.save_agent(self.agent_data, "nihao")
        
        # Load agent
        loaded = self.storage.load_agent("test_agent")
        
        # Verify data
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["name"], "test_agent")
        self.assertEqual(loaded["content"], self.agent_data["content"])  # JSON parsed
        self.assertEqual(loaded["date"], "2025-05-13")

    def test_save_and_load_workflow(self):
        """
        Test saving and loading a workflow, verifying data integrity and JSON parsing.
        """
        # Save workflow
        self.storage.save_workflow(self.workflow_data)
        
        # Load workflow
        loaded = self.storage.load_workflow("test_workflow")
        
        # Verify data
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["name"], "test_workflow")
        self.assertEqual(loaded["content"], self.workflow_data["content"])  # JSON parsed
        self.assertEqual(loaded["date"], "2025-05-13")

    def test_save_and_load_history(self):
        """
        Test saving and loading a history entry, verifying data integrity.
        """
        # Save history
        self.storage.save_history(self.history_data)
        
        # Load history
        loaded = self.storage.load_history("mem_001")
        
        # Verify data
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["memory_id"], "mem_001")
        self.assertEqual(loaded["old_memory"], "Initial content")
        self.assertEqual(loaded["new_memory"], "Updated content")
        self.assertEqual(loaded["event"], "update")
        self.assertEqual(loaded["created_at"], "2025-05-13T09:00:00")
        self.assertEqual(loaded["updated_at"], "2025-05-13T09:30:00")

    def test_load_non_existent_agent(self):
        """
        Test loading a non-existent agent returns None.
        """
        loaded = self.storage.load_agent("non_existent_agent")
        self.assertIsNone(loaded)

    def test_load_non_existent_workflow(self):
        """
        Test loading a non-existent workflow returns None.
        """
        loaded = self.storage.load_workflow("non_existent_workflow")
        self.assertIsNone(loaded)

    def test_load_non_existent_history(self):
        """
        Test loading a non-existent history entry returns None.
        """
        loaded = self.storage.load_history("non_existent_mem")
        self.assertIsNone(loaded)

    def test_save_invalid_agent(self):
        """
        Test saving an agent without a 'name' field raises ValueError.
        """
        invalid_data = {"content": {"role": "assistant"}, "date": "2025-05-13"}
        with self.assertRaises(ValueError):
            self.storage.save_agent(invalid_data)

    def test_save_invalid_workflow(self):
        """
        Test saving a workflow without a 'name' field raises ValueError.
        """
        invalid_data = {"content": {"steps": ["step1"]}, "date": "2025-05-13"}
        with self.assertRaises(ValueError):
            self.storage.save_workflow(invalid_data)

    def test_save_invalid_history(self):
        """
        Test saving a history entry without a 'memory_id' field raises ValueError.
        """
        invalid_data = {
            "old_memory": "Initial",
            "new_memory": "Updated",
            "event": "update"
        }
        with self.assertRaises(ValueError):
            self.storage.save_history(invalid_data)

    def test_remove_agent(self):
        """
        Test removing an agent and verify it's no longer loadable.
        """
        # Save and remove agent
        self.storage.save_agent(self.agent_data)
        self.storage.remove_agent("test_agent")
        
        # Verify agent is gone
        loaded = self.storage.load_agent("test_agent")
        self.assertIsNone(loaded)

    def test_remove_non_existent_agent(self):
        """
        Test removing a non-existent agent raises ValueError.
        """
        with self.assertRaises(ValueError):
            self.storage.remove_agent("non_existent_agent")

    def test_update_agent(self):
        """
        Test updating an existing agent's data.
        """
        # Save initial agent
        self.storage.save_agent(self.agent_data)
        
        # Update agent data
        updated_data = {
            "name": "test_agent",
            "content": {"role": "admin", "settings": {"active": False}},
            "date": "2025-05-14"
        }
        self.storage.save_agent(updated_data)
        
        # Load and verify updated data
        loaded = self.storage.load_agent("test_agent")
        self.assertEqual(loaded["content"], updated_data["content"])
        self.assertEqual(loaded["date"], "2025-05-14")

    def test_update_workflow(self):
        """
        Test updating an existing workflow's data.
        """
        # Save initial workflow
        self.storage.save_workflow(self.workflow_data)
        
        # Update workflow data
        updated_data = {
            "name": "test_workflow",
            "content": {"steps": ["step3"], "config": {"timeout": 60}},
            "date": "2025-05-14"
        }
        self.storage.save_workflow(updated_data)
        
        # Load and verify updated data
        loaded = self.storage.load_workflow("test_workflow")
        self.assertEqual(loaded["content"], updated_data["content"])
        self.assertEqual(loaded["date"], "2025-05-14")

    def test_update_history(self):
        """
        Test updating an existing history entry.
        """
        # Save initial history
        self.storage.save_history(self.history_data)
        
        # Update history data
        updated_data = {
            "memory_id": "mem_001",
            "old_memory": "Initial content",
            "new_memory": "Further updated content",
            "event": "modify",
            "created_at": "2025-05-13T09:00:00",
            "updated_at": "2025-05-13T10:00:00"
        }
        self.storage.save_history(updated_data)
        
        # Load and verify updated data
        loaded = self.storage.load_history("mem_001")
        self.assertEqual(loaded["new_memory"], "Further updated content")
        self.assertEqual(loaded["event"], "modify")
        self.assertEqual(loaded["updated_at"], "2025-05-13T10:00:00")

    def test_bulk_save_and_load(self):
        """
        Test saving multiple records to all tables and loading them.
        """
        # Prepare bulk data
        agent_data2 = {
            "name": "test_agent2",
            "content": {"role": "user", "settings": {"active": True}},
            "date": "2025-05-13"
        }
        workflow_data2 = {
            "name": "test_workflow2",
            "content": {"steps": ["stepA", "stepB"], "config": {"timeout": 45}},
            "date": "2025-05-13"
        }
        history_data2 = {
            "memory_id": "mem_002",
            "old_memory": "Old content",
            "new_memory": "New content",
            "event": "create",
            "created_at": "2025-05-13T10:00:00",
            "updated_at": "2025-05-13T10:00:00"
        }
        
        # Save bulk data
        bulk_data = {
            TableType.store_agent.value: [self.agent_data, agent_data2],
            TableType.store_workflow.value: [self.workflow_data, workflow_data2],
            TableType.store_history.value: [self.history_data, history_data2]
        }
        self.storage.save(bulk_data)
        
        # Load all data
        all_data = self.storage.load()
        
        # Verify data presence
        self.assertIn(TableType.store_agent.value, all_data)
        self.assertIn(TableType.store_workflow.value, all_data)
        self.assertIn(TableType.store_history.value, all_data)
        self.assertEqual(len(all_data[TableType.store_agent.value]), 2)
        self.assertEqual(len(all_data[TableType.store_workflow.value]), 2)
        self.assertEqual(len(all_data[TableType.store_history.value]), 2)
        
        # Verify specific records
        agent_names = [record["name"] for record in all_data[TableType.store_agent.value]]
        self.assertIn("test_agent", agent_names)
        self.assertIn("test_agent2", agent_names)
        
        workflow_names = [record["name"] for record in all_data[TableType.store_workflow.value]]
        self.assertIn("test_workflow", workflow_names)
        self.assertIn("test_workflow2", workflow_names)
        
        history_ids = [record["memory_id"] for record in all_data[TableType.store_history.value]]
        self.assertIn("mem_001", history_ids)
        self.assertIn("mem_002", history_ids)

    def test_save_invalid_table(self):
        """
        Test saving data to an unknown table raises ValueError.
        """
        invalid_data = {"unknown_table": [self.agent_data]}
        with self.assertRaises(ValueError):
            self.storage.save(invalid_data)

    def tearDown(self):
        """
        Clean up by closing the database connection.
        """
        self.storage.storageDB.connection.close()

if __name__ == "__main__":
    unittest.main()