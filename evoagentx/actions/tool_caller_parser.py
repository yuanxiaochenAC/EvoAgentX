import json
from typing import Dict, Any, Optional
from pydantic import Field, field_validator

from ..core.logging import logger
from .action import ActionOutput


class ToolCallerParser(ActionOutput):
    action: str = Field(description="Type of action: 'tool_call' or 'direct_answer'")
    content: Dict[str, Any] = Field(description="Content of the response based on action type")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        if v not in ["tool_call", "direct_answer"]:
            raise ValueError(f"Invalid action: {v}. Must be 'tool_call' or 'direct_answer'")
        return v
    
    @classmethod
    def from_str(cls, content: str, **kwargs):
        """
        Parse LLM output string into a structured format
        
        Args:
            content (str): The string output from the LLM
            
        Returns:
            ToolCallerParser: The parsed output
        """
        try:
            # Extract the JSON content if it's wrapped in markdown code blocks
            if "```json" in content:
                start_index = content.find("```json") + 7
                end_index = content.find("```", start_index)
                if end_index != -1:
                    json_str = content[start_index:end_index].strip()
                else:
                    json_str = content[start_index:].strip()
            elif "```" in content:
                start_index = content.find("```") + 3
                end_index = content.find("```", start_index)
                if end_index != -1:
                    json_str = content[start_index:end_index].strip()
                else:
                    json_str = content[start_index:].strip()
            else:
                json_str = content.strip()
            
            # Parse the JSON string
            data = json.loads(json_str)
            
            # Create the parser instance
            return cls(
                action=data.get("action", ""),
                content=data.get("content", {})
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            # Fallback to direct answer if parsing fails
            return cls(
                action="direct_answer",
                content={"answer": content, "reasoning": "Parsing error, treating as direct answer"}
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing tool caller output: {e}")
            return cls(
                action="direct_answer",
                content={"answer": content, "reasoning": f"Error parsing: {str(e)}"}
            )
    
    def get_tool_call_info(self) -> Optional[Dict[str, Any]]:
        """
        Get tool call information if action is tool_call
        
        Returns:
            Dict containing tool_name and parameters, or None if not a tool call
        """
        if self.action != "tool_call":
            return None
        
        return {
            "tool_name": self.content.get("tool_name", ""),
            "parameters": self.content.get("parameters", {}),
            "reasoning": self.content.get("reasoning", "")
        }
    
    def get_direct_answer(self) -> Optional[str]:
        """
        Get the direct answer if action is direct_answer
        
        Returns:
            The answer string or None if not a direct answer
        """
        if self.action != "direct_answer":
            return None
        
        return self.content.get("answer", "")
    
    def to_str(self) -> str:
        """
        Convert the parsed output back to a string representation
        """
        return json.dumps({"action": self.action, "content": self.content}, indent=2)