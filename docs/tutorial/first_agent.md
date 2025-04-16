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