import os
import json
from typing import Dict, Any

from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.models.openai_model import OpenAILLM

# Simple weather tool definitions
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def test_openai_tools():
    # Get OpenAI API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY environment variable not set. Skipping test.")
        return
    
    # Create a subclass that implements tool execution
    class WeatherLLM(OpenAILLM):
        def _execute_tool_call(self, function_name: str, arguments: Dict[str, Any]) -> Any:
            if function_name == "get_current_weather":
                location = arguments.get("location", "Unknown")
                unit = arguments.get("unit", "celsius")
                
                # Return simulated weather data
                return {
                    "location": location,
                    "temperature": 22 if unit == "celsius" else 72,
                    "unit": unit,
                    "condition": "Sunny",
                    "humidity": 60
                }
            else:
                return {"error": f"Unknown function: {function_name}"}
    
    # Create model config with tools
    llm_config = OpenAILLMConfig(
        llm_type="OpenAILLM",
        model="gpt-3.5-turbo",
        openai_key=openai_api_key,
        temperature=0.7,
        max_tokens=500,
        tools=WEATHER_TOOLS,
        tool_choice="auto"
    )
    
    # Initialize the model with tool handling
    llm = WeatherLLM(config=llm_config)
    
    # Prepare messages to invoke tool
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can check the weather."},
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
    
    # Generate a response
    print("\nSending request to OpenAI API...")
    response = llm.single_generate(messages=messages)
    
    print("\nFinal response:")
    print(response)

if __name__ == "__main__":
    test_openai_tools() 