# EvoAgentX

<p align="center">
  <em>An automatic agentic workflow generation and evolving framework.</em>
</p>

## 🚀 Introduction

EvoAgentX is a powerful framework designed to automate the generation, optimization, and execution of agentic workflows. By leveraging large language models, EvoAgentX enables developers to create complex multi-agent systems that can evolve over time to solve increasingly difficult tasks.

## ✨ Key Features

- **Automatic Workflow Generation**: Create complex agent workflows with a simple goal description
- **Evolutionary Optimization**: Improve workflows through continuous evolution and adaptation
- **Flexible Agent Management**: Define custom agents or use built-in templates
- **Workflow Visualization**: Visualize and inspect your agent workflows
- **Persistent Storage**: Save and reuse successful workflows and agents
- **Evaluation Framework**: Measure and compare workflow performance

## 🔍 How It Works

EvoAgentX uses a modular architecture with the following core components:

1. **Workflow Generator**: Creates optimal agent workflows based on your goals
2. **Agent Manager**: Handles agent creation, configuration, and deployment
3. **Workflow Executor**: Runs workflows efficiently with proper agent communication
4. **Evaluators**: Provides performance metrics and improvement suggestions
5. **Optimizers**: Evolves workflows to enhance performance over time

## 🛠️ Quick Installation

```bash
# Create environment
conda create -n agent python=3.10
conda activate agent

# Install package
pip install -r requirements.txt
```

## 📝 Basic Usage

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import AgentManager
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow

# Configure your language model
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini", 
    openai_key=YOUR_API_KEY, 
    stream=True
)
model = OpenAILLM(config=openai_config)

# Set up agent manager and workflow generator
agent_manager = AgentManager()
wf_generator = WorkFlowGenerator(llm=model)

# Generate workflow & agents
workflow_graph = wf_generator.generate_workflow(
    goal="Generate a python code for greedy snake game"
)

# Visualize workflow (optional)
workflow_graph.display()

# Execute workflow
workflow = WorkFlow(
    graph=workflow_graph, 
    agent_manager=agent_manager, 
    llm=model
)
output = workflow.execute()
```

## 📚 Documentation

Learn more about EvoAgentX and how to use it effectively:

- **API Reference**: Detailed documentation of all modules and functions
- **Tutorials**: Step-by-step guides for common use cases
- **Examples**: Sample code and recipes for different scenarios

## 👥 Community

- **Discord**: Join our [Discord Channel](https://discord.gg/q5hBjHVz) for discussions and support
- **GitHub**: Contribute to the project on [GitHub](https://github.com/clayxai/EvoAgentX)
- **Email**: Contact us at clayx.ai.co@gmail.com

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/clayxai/EvoAgentX/blob/main/CONTRIBUTING.md) for more details.
