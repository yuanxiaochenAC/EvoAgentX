PDF_AGENT_SUGGESTION = """
    It would be better to include a short summary on the resume at the beginning of the output.
    You should use the tool_calling action to interact with the website to find real job opportunities.
    I already give you the path in this goal, and there is no other inputs for you. You should try to find a way to read it.
    The output should be well-formatted.
    You should return every detailed information you collected from the website, inclduing job title, company name, location, detailed job description, detailed job requirements, salary, job posting link, etc.
    """

def formulate_goal(goal):
    return f"""
    {goal}
    {PDF_AGENT_SUGGESTION}
    """