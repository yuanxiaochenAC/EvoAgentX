from typing import Any, Dict, List

from evoagentx.tools.tool import Tool


class GraphExtract(Tool):
    
    def __init__(
        self, 
        name: str = "GraphExtract",
        max_paths_per_chunk: int = 5,
        **kwargs
    ):
        """Initialize the Graph Extract tool/

        Args:
            name (str): The name of the tool.
            max_paths_per_chunk (int): The maximum number of paths per chunk.
        """
        super().__init__(name=name, max_paths_per_chunk=max_paths_per_chunk, **kwargs)

    def extract_entity(self, entities: List[Dict[str, str]]):
        """
        Extract entities from the given text.

        Args:
            entities (List[Dict[str, str]]): A list of dictionaries containing entity information.
                Each dictionary should have keys 'entity' and 'entity_type' with corresponding values.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing extracted entity information.
                Each dictionary has keys 'entity' and 'entity_type' with corresponding values.
        """
        return [{entity["entity"]: entity["entity_type"]} for entity in entities]

    def extract_relationship(self, entities: List[Dict[str, str]]):
        """
        Extract relationships between entities from the given text.

        Args:
            entities (List[Dict[str, str]]): A list of dictionaries containing entity relationship information.
                Each dictionary should have keys 'src_entity', 'relationship', and 'dst_entity' with corresponding values.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing extracted relationship information.
                Each dictionary has keys 'src_entity', 'relationship', and 'dst_entity' with corresponding values.
        """
        for item in entities:
            item["src_entity"] = item["src_entity"].lower().strip().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().strip().replace(" ", "_")
            item["dst_entity"] = item["dst_entity"].lower().strip().replace(" ", "_")
        return entities

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "extract_entity",
                    "description": "Extract entities and their types from the given text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "entity": {"type": "string", "description": "The name or identifier of the entity."},
                                        "entity_type": {"type": "string", "description": "The type or category of the entity."},
                                    },
                                    "required": ["entity", "entity_type"],
                                },
                                "description": "An array of entities with their types.",
                            }
                        },
                        "required": ["entities"],
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "extract_relationship",
                    "description": "Extract relationships between entities based on the given text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "src_entity": {"type": "string", "description": "The source entity of the relationship."},
                                        "relationship": {"type": "string", "description": "The relationship between the source and destination entities."},
                                        "dst_entity": {"type": "string", "description": "The destination entity of the relationship."},
                                    },
                                    "required": ["src_entity", "relationship", "dst_entity"],
                                }
                            }
                        },
                        "required": ["entities"],
                    }
                }
            }
        ]
    
    def get_tools(self):
        return [self.extract_entity, self.extract_relationship]
    
