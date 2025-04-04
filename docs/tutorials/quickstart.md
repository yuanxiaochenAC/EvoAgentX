# EvoAgentX Quickstart Guide

This quickstart guide will walk you through the fundamental steps to get up and running with EvoAgentX. By the end of this tutorial, you'll understand how to:

1. Configure your API keys and settings
2. Build your first agent
3. Execute agents to perform tasks
4. Manually build custom workflows

## Prerequisites

Before starting this tutorial, make sure you have:
- Installed EvoAgentX (see the [Installation Guide](./installation.md))
- Access to an OpenAI API key or other supported LLM provider
- Basic understanding of Python

## 1. Key Configuration

The first step is to configure your API keys and environment settings.

### Environment Variable Setup

The recommended way to manage your API keys is through environment variables:

```bash
# Create a .env file based on the example
cp .env.example .env

# Edit the .env file with your API keys
```

Your `.env` file should include at least the following:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Programmatic Configuration

You can also configure your settings programmatically in your Python code:

```python
import os
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# Method 1: Set environment variable in code
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Method 2: Pass API key directly to the config
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",  # Choose the model you want to use
    openai_key="your_openai_api_key_here",  # Direct API key configuration
    stream=True,  # Enable streaming for real-time responses
    temperature=0.7,  # Control creativity (0.0 = deterministic, 1.0 = creative)
    max_tokens=1000  # Maximum response length
)

# Initialize the language model
llm = OpenAILLM(config=openai_config)
```

## 2. Building Your First Agent

In EvoAgentX, agents are the building blocks that perform specific tasks. Let's create a simple agent:

### Basic Agent Creation

```python
from evoagentx.agents import Agent, AgentManager

# Create an agent manager to handle your agents
agent_manager = AgentManager()

# Define a simple agent
simple_agent = Agent(
    name="CodeWriter",
    description="Writes clean Python code based on requirements",
    system_prompt="You are an expert Python developer specialized in writing clean, efficient code. Respond only with code and brief explanations when necessary."
)

# Add the agent to the manager
agent_manager.add_agent(simple_agent)
```

### Agent with Custom Tools

Agents can use tools to extend their capabilities:

```python
from evoagentx.tools import Tool

# Define a simple calculator tool
def calculator(expression):
    """Evaluate a mathematical expression"""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"

# Create a tool from the function
calc_tool = Tool(
    name="calculator",
    description="Evaluates mathematical expressions",
    function=calculator
)

# Create an agent with the tool
math_agent = Agent(
    name="MathSolver",
    description="Solves mathematical problems",
    system_prompt="You are a mathematics expert. Use the calculator tool to solve math problems.",
    tools=[calc_tool]
)

# Add to the agent manager
agent_manager.add_agent(math_agent)
```

## 3. Agent Execution

Once you've created your agents, you can execute them to perform tasks:

### Running a Single Agent

```python
# Get the agent from the manager
code_writer = agent_manager.get_agent("CodeWriter")

# Execute the agent with a specific task
response = code_writer.run(
    user_input="Write a function to find the nth Fibonacci number using dynamic programming."
)

# Print the response
print(response)
```

### Agent with Memory

Agents can maintain memory of past interactions:

```python
from evoagentx.memory import ConversationMemory

# Create an agent with memory
assistant = Agent(
    name="Assistant",
    description="A helpful assistant that remembers past interactions",
    system_prompt="You are a helpful assistant. Remember details from previous interactions.",
    memory=ConversationMemory()  # Add conversation memory
)

# First interaction
response1 = assistant.run("My name is Alice.")
print(response1)

# Second interaction (agent will remember the name)
response2 = assistant.run("What's my name?")
print(response2)  # Should output something like "Your name is Alice."
```

## 4. Building Workflows Manually

Workflows allow multiple agents to collaborate on complex tasks. Here's how to build one manually:

### Creating a Simple Workflow

```python
from evoagentx.workflow import WorkFlowGraph, WorkFlow, Node, Edge

# Create a workflow graph
workflow_graph = WorkFlowGraph()

# Create nodes for our agents
research_node = Node(agent_name="Researcher", node_id="research", description="Researches the topic")
code_node = Node(agent_name="CodeWriter", node_id="code", description="Writes code based on research")
review_node = Node(agent_name="Reviewer", node_id="review", description="Reviews the code")

# Add nodes to the graph
workflow_graph.add_node(research_node)
workflow_graph.add_node(code_node)
workflow_graph.add_node(review_node)

# Connect the nodes with edges
workflow_graph.add_edge(Edge(source_id="research", target_id="code"))
workflow_graph.add_edge(Edge(source_id="code", target_id="review"))

# Create the workflow
workflow = WorkFlow(
    graph=workflow_graph,
    agent_manager=agent_manager,
    llm=llm
)

# Execute the workflow with an initial input
result = workflow.execute(input_data="Create a weather forecast application using OpenWeatherMap API")
print(result)
```

### Visualizing Your Workflow

You can visualize the workflow to better understand its structure:

```python
# Display the workflow
workflow_graph.display()

# Save the workflow for future use
workflow_graph.save_module("my_workflow.json")
```

### Loading a Saved Workflow

You can load a previously saved workflow:

```python
# Load a workflow from a file
loaded_workflow = WorkFlowGraph.from_file("my_workflow.json")

# Create a new workflow instance with the loaded graph
new_workflow = WorkFlow(
    graph=loaded_workflow,
    agent_manager=agent_manager,
    llm=llm
)

# Execute the loaded workflow
result = new_workflow.execute(input_data="Create a to-do list application")
```

## Putting It All Together

Here's a complete example that combines all the concepts we've covered:

```python
import os
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import Agent, AgentManager
from evoagentx.workflow import WorkFlowGraph, WorkFlow, Node, Edge
from evoagentx.memory import ConversationMemory

# 1. Configure the language model
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",
    openai_key=os.environ.get("OPENAI_API_KEY"),
    stream=True
)
llm = OpenAILLM(config=openai_config)

# 2. Create an agent manager
agent_manager = AgentManager()

# 3. Define your agents
researcher = Agent(
    name="Researcher",
    description="Researches topics and provides information",
    system_prompt="You are an expert researcher. Find relevant information on the given topic.",
    memory=ConversationMemory()
)

planner = Agent(
    name="Planner",
    description="Plans how to implement solutions",
    system_prompt="You are a project planner. Create detailed plans for implementing solutions."
)

coder = Agent(
    name="Coder",
    description="Writes code based on plans",
    system_prompt="You are an expert programmer. Write clean, efficient code based on the provided plans."
)

reviewer = Agent(
    name="Reviewer",
    description="Reviews code for quality and correctness",
    system_prompt="You are a code reviewer. Evaluate the code for bugs, efficiency, and best practices."
)

# 4. Add agents to the manager
agent_manager.add_agent(researcher)
agent_manager.add_agent(planner)
agent_manager.add_agent(coder)
agent_manager.add_agent(reviewer)

# 5. Build a workflow graph
workflow_graph = WorkFlowGraph()

# Add nodes
workflow_graph.add_node(Node(agent_name="Researcher", node_id="research", description="Research phase"))
workflow_graph.add_node(Node(agent_name="Planner", node_id="planning", description="Planning phase"))
workflow_graph.add_node(Node(agent_name="Coder", node_id="coding", description="Implementation phase"))
workflow_graph.add_node(Node(agent_name="Reviewer", node_id="review", description="Review phase"))

# Connect nodes
workflow_graph.add_edge(Edge(source_id="research", target_id="planning"))
workflow_graph.add_edge(Edge(source_id="planning", target_id="coding"))
workflow_graph.add_edge(Edge(source_id="coding", target_id="review"))

# 6. Create the workflow
workflow = WorkFlow(
    graph=workflow_graph,
    agent_manager=agent_manager,
    llm=llm
)

# 7. Execute the workflow
project_goal = "Create a web scraper that extracts product information from e-commerce websites"
result = workflow.execute(input_data=project_goal)

print("\n=== Final Result ===")
print(result)
```

## Next Steps

Now that you've learned the basics of EvoAgentX, you can:

- Explore the [API documentation](../api.md) to learn about more advanced features
- Try building more complex workflows with conditional branching and loops
- Experiment with different agent configurations and tool integrations
- Check out the [example projects](../examples/index.md) for inspiration

For more advanced topics, see our guides on:
- [Advanced Workflow Patterns](../advanced/workflow_patterns.md)
- [Custom Tool Development](../advanced/custom_tools.md)
- [Memory Systems](../advanced/memory_systems.md)
- [Evolutionary Optimization](../advanced/evolutionary_optimization.md)
