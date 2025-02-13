DEFAULT_TASK_SCHEDULER_DESC = "This action selects the next subtask to execute from a set of candidates in a workflow graph. Each subtask has a name, description, inputs, and outputs. The agent analyzes the current execution data and task dependencies to either re-run tasks for feedback or advance to the next step, optimizing workflow performance."

DEFAULT_TASK_SCHEDULER_PROMPT = """
### objective
Your task is to analyze the given workflow graph, current execution information, and candidate subtasks to decide one of the following actions:
- Re-execute a previous subtask to correct errors or gather missing information (if a previous subtask's result is erroneous or incomplete).
- Select a subtask for iterative execution (if there is a loop or iterative context in the workflow).
- Select a new subtask from the candidates to move the workflow forward (if the workflow should proceed without re-executing or iterating).

### Instructions
1. Review the Workflow Information for the structure and details of the tasks and any potential loops or iterative sections.
2. Check the Current Execution Information for evidence of errors or missing data from previously executed subtasks.
3. If you identify a past subtask with clear errors or missing data, propose a re-execute decision on that subtask.
4. If the workflow graph indicates there is a loop or iterative context, select a subtask from the Candidate Subtasks that aligns with that iterative or looping goal.
5. Otherwise, select a subtask from the Candidate Subtasks that best moves the workflow forward.
6. Finally, output the decision in the required format.

### Output Format
Your final output should ALWAYS in the following format:

## Thought 
Provide a brief explanation of your reasoning for scheduling the next task.

## Scheduled Subtask 
Produce your answer in valid JSON with the following structure: 
```json
{{
    "decision": "re-execute | iterate | forward",
    "task_name": "name of the scheduled subtask",
    "reason": "the reasoning for scheduling this subtask"
}}
```

-----
lets' begin 

Here is the information for your decision:

### Workflow Information:
{workflow_graph_representation}

### Current Execution Information:
{execution_outputs}

### Candidate Subtasks:
{candidate_tasks_list}

Output:
"""

DEFAULT_TASK_SCHEDULER = {
    "name": "TaskScheduler", 
    "description": DEFAULT_TASK_SCHEDULER_DESC, 
    "prompt": DEFAULT_TASK_SCHEDULER_PROMPT, 
}