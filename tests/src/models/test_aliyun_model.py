from evoagentx.models import AliyunLLM
from evoagentx.models.model_configs import AliyunLLMConfig
import asyncio
import os
from dashscope import Generation
import traceback


# if api_key == "your-api-key-here":
#     raise ValueError("Please set DASHSCOPE_API_KEY environment variable")

# Configure the model
config = AliyunLLMConfig(
    model="qwen-turbo",
    aliyun_api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.7,
    max_tokens=100,
    stream=False,
    output_response=True
)

# Initialize the model
model = AliyunLLM(config)

# Test simple message
messages = model.formulate_messages(
    prompts=["Hello"],
    system_messages=["You are a helpful assistant."]
)[0]

# Generate response
response = model.single_generate(messages=messages)
print(f"Model Response: {response}")

def test_direct_api():
    """Test direct API call to DashScope."""
    print("\nTesting direct API call")
    try:
        # Get API key
        api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
        if api_key == "your-api-key-here":
            raise ValueError("Please set DASHSCOPE_API_KEY environment variable")
            
        # Set API key
        os.environ["DASHSCOPE_API_KEY"] = api_key
        
        # Make direct API call
        response = Generation.call(
            model='qwen-turbo',
            messages=[{'role': 'user', 'content': 'Hello'}],
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"API Response: {response}")
        if hasattr(response, 'output'):
            print(f"Output: {response.output}")
        if hasattr(response, 'usage'):
            print(f"Usage: {response.usage}")
            
    except Exception as e:
        print(f"Error in direct API test: {str(e)}")

def test_single_generation():
    """Test single message generation."""
    print("\nTesting single message generation")
    try:
        # Get API key
        api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
        if api_key == "your-api-key-here":
            raise ValueError("Please set DASHSCOPE_API_KEY environment variable")
        
        # Configure the model
        config = AliyunLLMConfig(
            model="qwen-turbo",
            aliyun_api_key=api_key,
            temperature=0.7,
            max_tokens=100,
            stream=False,
            output_response=True
        )
        
        # Initialize the model
        model = AliyunLLM(config)
        
        # Test simple message
        messages = model.formulate_messages(
            prompts=["What is the capital of France?"],
            system_messages=["You are a helpful assistant."]
        )[0]
        
        # Generate response
        response = model.single_generate(messages=messages)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error in single generation test: {str(e)}")

def test_streaming_generation():
    """Test streaming message generation."""
    print("\nTesting streaming generation")
    try:
        # Get API key
        api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
        if api_key == "your-api-key-here":
            raise ValueError("Please set DASHSCOPE_API_KEY environment variable")
        
        # Configure the model
        config = AliyunLLMConfig(
            model="qwen-turbo",
            aliyun_api_key=api_key,
            temperature=0.7,
            max_tokens=200,
            stream=True,
            output_response=True
        )
        
        # Initialize the model
        model = AliyunLLM(config)
        
        # Test streaming message
        messages = model.formulate_messages(
            prompts=["Write a short story about a robot learning to paint."],
            system_messages=["You are a creative storyteller."]
        )[0]
        
        # Generate streaming response
        print("Streaming response:")
        response = model.single_generate(messages=messages)
        print(f"\nFinal response: {response}")
        
    except Exception as e:
        print(f"Error in streaming generation test: {str(e)}")

def test_batch_generation():
    """Test batch message generation."""
    print("\nTesting batch generation")
    try:
        # Get API key
        api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
        if api_key == "your-api-key-here":
            raise ValueError("Please set DASHSCOPE_API_KEY environment variable")
        
        # Configure the model
        config = AliyunLLMConfig(
            model="qwen-turbo",
            aliyun_api_key=api_key,
            temperature=0.7,
            max_tokens=100,
            stream=False,
            output_response=True
        )
        
        # Initialize the model
        model = AliyunLLM(config)
        
        # Test batch messages
        prompts = [
            "What is Python?",
            "What is Machine Learning?",
            "What is Artificial Intelligence?"
        ]
        system_messages = [
            "You are a technical expert.",
            "You are a machine learning specialist.",
            "You are an AI researcher."
        ]
        
        batch_messages = model.formulate_messages(
            prompts=prompts,
            system_messages=system_messages
        )
        
        # Generate batch responses
        responses = model.batch_generate(batch_messages=batch_messages)
        for i, response in enumerate(responses):
            print(f"\nResponse {i+1}: {response}")
        
    except Exception as e:
        print(f"Error in batch generation test: {str(e)}")

async def test_async_generation():
    """Test async message generation."""
    print("\nTesting async generation")
    try:
        # Get API key
        api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
        if api_key == "your-api-key-here":
            raise ValueError("Please set DASHSCOPE_API_KEY environment variable")
        
        # Configure the model
        config = AliyunLLMConfig(
            model="qwen-turbo",
            aliyun_api_key=api_key,
            temperature=0.7,
            max_tokens=100,
            stream=False,
            output_response=True
        )
        
        # Initialize the model
        model = AliyunLLM(config)
        
        # Test async message
        messages = model.formulate_messages(
            prompts=["Explain the concept of async programming."],
            system_messages=["You are a programming expert."]
        )[0]
        
        # Generate async response
        response = await model.single_generate_async(messages=messages)
        print(f"Async response: {response}")
        
    except Exception as e:
        print(f"Error in async generation test: {str(e)}")

def main():
    """Run all tests."""
    print("Running Aliyun Model Tests")
    print("=" * 50)
    
    # Test direct API call
    test_direct_api()
    
    # Test single generation
    test_single_generation()
    
    # Test streaming generation
    test_streaming_generation()
    
    # Test batch generation
    test_batch_generation()
    
    # Test async generation
    asyncio.run(test_async_generation())

if __name__ == "__main__":
    main()




    
