WORKFLOW_GENERATION_PROMPT = """
## Topic
You are expected to generate a workflow for the given topic:
{goal}

## Generated Workflow
### Generated Workflow Inputs Requirements
Your workflow should have the following inputs:
{inputs}

### Generated Workflow Outputs Requirements
Your workflow should have the following outputs:
{outputs}

"""

TASK_INFO_PROMPT_SUDO = """
## Workflow Information
You are expected to generate a workflow information based on the given goal:
{goal}

## Output
- Your output should be a JSON object contain the following fields:
    - workflow_name: string
    - workflow_description: string
    - workflow_inputs: dictionary, fixed to only have "goal" as the input
    - workflow_outputs: dictionary, fixed to only have "workflow_output" as the output
- Here is an example of the output:
{{
    "workflow_name": "Market Analysis Workflow",
    "workflow_description": "This workflow is used to analyze the market and generate a report.",
    "workflow_inputs": [
        {{"name": "goal", "type": "string", "description": "The original full programming task requirements.", "required": True}},
    ],
    "workflow_outputs": [
        {{"name": "workflow_output", "type": "string", "description": "The workflow output.", "required": True}},
    ]
}}

## Additional Information
You might be given additional information or instructions about the project. You should follow them as well.
{additional_info}

"""

CONNECTION_INSTRUCTION_PROMPT = """# Project Access Instructions

## Project Information
- **Project ID**: {project_id}
- **Server URL**: {public_url}

## API Endpoints

### 1. Execute Workflow
Run the workflow with your specific input:

```bash
curl -X POST {public_url}/workflow/execute \\
  -H "Content-Type: application/json" \\
  -d '{{
    "project_id": "{project_id}",
    "inputs": {{
      "goal": "Analyze the price and trend for company Apple"
    }},
    "llm_config": {{
      "model": "gpt-4o-mini",
      "openai_key": "your_openai_api_key"
    }}
  }}'
```

**Expected Response:**
```json
{{
  "success": true,
  "execution_result": {{
    "status": "completed",
    "message": "# Comprehensive Report: AAPL Stock Performance Analysis\\n\\n### Current Price Metrics\\n- Stock Symbol: AAPL\\n- Latest Price: $175.30..."
  }},
  "message": "Workflow executed successfully"
}}
```

### 2. Check Project Status
Monitor your project status:

```bash
curl -X GET {public_url}/project/{project_id}/status
```

## Quick Start Example

1. **Generate Workflow** (one-time setup):
   ```bash
   curl -X POST {public_url}/workflow/generate \\
     -H "Content-Type: application/json" \\
     -d '{{"project_id": "{project_id}"}}'
   ```

2. **Execute Analysis**:
   ```bash
   curl -X POST {public_url}/workflow/execute \\
     -H "Content-Type: application/json" \\
     -d '{{
       "project_id": "{project_id}",
       "inputs": {{"goal": "Analyze Tesla stock performance"}}
     }}'
   ```

## Input Examples
- "Analyze the price and trend for company Apple"
- "Provide stock analysis for Microsoft with technical indicators"  
- "Generate investment report for Google including market sentiment"

## Notes
- LLM config is optional, dont use it unless you have related information
- The workflow generates comprehensive stock analysis reports
- Each execution can analyze different companies by changing the input goal
- Generated workflows are reusable for multiple executions

"""

TASK_INFO_PROMPT = """

## Workflow Information
You are expected to generate a workflow information based on the given goal:
{goal}

## Output
- Your output should be a JSON object contain the following fields:
    - workflow_name: string
    - workflow_description: string
    - workflow_inputs: dictionary
    - workflow_outputs: dictionary, for the most of the time, one output is enough
- Here is an example of the output:
{{
    "workflow_name": "Market Analysis Workflow",
    "workflow_description": "This workflow is used to analyze the market and generate a report.",
    "workflow_inputs": [
        {{"name": "goal", "type": "string", "description": "The original full programming task requirements.", "required": True}},
        {{"name": "personal_details", "type": "array", "description": "An array of job opportunities obtained from the search.", "required": True}}
    ],
    "workflow_outputs": [
        {{"name": "resume_summary", "type": "string", "description": "A summary of the resume.", "required": True}},
        {{"name": "job_opportunities", "type": "array", "description": "An array of job opportunities obtained from the search.", "required": True}}
    ]
}}

## Additional Information
You might be given additional information or instructions about the project. You should follow them as well.
{additional_info}

"""







