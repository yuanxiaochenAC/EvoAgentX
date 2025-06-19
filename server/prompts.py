WORKFLOW_GENERATION_PROMPT = """
## Role
You are a data processing workflow expert. You create workflows that analyze data and answer queries - NOT workflows that generate websites or code.

## Context
{goal}

A development team has already built a website that collects user inputs. Your workflow receives these processed inputs and returns analysis results. The development team handles all frontend/backend code - your job is purely data processing and analysis.

## Workflow Requirements
**Inputs:** {inputs}
**Outputs:** {outputs} (must be a markdown formatted analysis/report)

Generate a workflow that processes the input data and provides analytical insights, research, or answers to user queries.
"""

TASK_INFO_PROMPT_SUDO = """
## Role
Generate workflow information for a data processing service. The workflow analyzes data and answers queries - it does NOT generate websites or code.

## Context
{goal}

A development team builds websites that collect user data. Your workflow receives this processed data and returns analytical results. You handle data processing and analysis only.

## Output Format
JSON object with these fields:
- workflow_name: string
- workflow_description: string  
- workflow_inputs: array (fixed format with "goal" input)
- workflow_outputs: array (fixed format with "workflow_output" output)

## Example
{{
    "workflow_name": "Market Analysis Workflow",
    "workflow_description": "Analyzes market data and generates analytical reports",
    "workflow_inputs": [
        {{"name": "goal", "type": "string", "description": "User query and data to analyze", "required": true}}
    ],
    "workflow_outputs": [
        {{"name": "workflow_output", "type": "string", "description": "Markdown formatted analysis report", "required": true}}
    ]
}}

## Additional Instructions
{additional_info}
"""

CONNECTION_INSTRUCTION_PROMPT = """
Call Alex to complete the following tasks with the following instruction. The final result should be in Markdown format.

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

## Notes
- Generated workflows are reusable for multiple executions

# Workflow Information / Guide
While using the workflow, you should strictly follow the following instructions:

## Basic Information
- **workflow_name**: {workflow_name}
- **workflow_description**: {workflow_description}

## Workflow Inputs Format
- **workflow_inputs**: {workflow_inputs}

## Workflow Outputs Format
- **workflow_outputs**: {workflow_outputs}


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
    "workflow_inputs": {{"target": "The target of the workflow", "personal_details": "The personal details of the user."}},
    "workflow_outputs": {{"resume_summary": "A summary of the resume.", "job_opportunities": "An array of job opportunities obtained from the search."}}
}}

## Additional Information
You might be given additional information or instructions about the project. You should follow them as well.
{additional_info}

"""







