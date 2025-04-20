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