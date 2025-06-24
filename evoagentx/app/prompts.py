WORKFLOW_GENERATION_PROMPT = """
## Role
You are a data processing workflow expert. You create workflows that analyze data and answer queries - NOT workflows that generate websites or code.

## Context
{goal}

A development team has already built a website that collects user inputs. Your workflow receives these processed inputs and returns analysis results. The development team handles all frontend/backend code - your job is purely data processing and analysis.

## Workflow Requirements
**Inputs:** {inputs_format}
**Outputs:** {outputs_format}

Generate a workflow that processes the input data and provides analytical insights, research, or answers to user queries.
"""

TASK_INFO_PROMPT = """
## Role
Generate detailed workflow specifications for a data processing service. The workflow analyzes data and answers queries - it does NOT generate websites or code.

## Context
{goal}

A development team builds applications that collect user data. Your workflow receives this processed data and returns analytical results. You handle data processing and analysis only.

## Output Format
JSON object with these fields:
- workflow_name: string (descriptive name for the workflow)
- workflow_description: string (detailed description of what the workflow does)
- inputs_format: object with key-value pairs where keys are input names and values are descriptions
- outputs_format: object with key-value pairs where keys are output names and values are descriptions

## Example
{{
    "workflow_name": "Market Analysis Workflow",
    "workflow_description": "Analyzes market data, stock performance, and generates comprehensive analytical reports with insights and recommendations",
    "inputs_format": {{
        "target_company": "The company or stock symbol to analyze",
        "analysis_period": "Time period for the analysis (e.g., last 6 months, 1 year)",
        "analysis_type": "Type of analysis required (technical, fundamental, or comprehensive)"
    }},
    "outputs_format": {{
        "analysis_report": "Comprehensive markdown formatted analysis report with insights, trends, and recommendations",
        "summary": "Executive summary of key findings and recommendations"
    }}
}}

## Additional Instructions
{additional_info}

## Guidelines
- Focus on data processing and analytical workflows
- Ensure inputs are specific and actionable
- Outputs should be useful for decision-making
- Keep workflow scope manageable and focused
- Avoid any code generation or web development tasks
"""
