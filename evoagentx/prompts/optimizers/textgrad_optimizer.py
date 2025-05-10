
from textgrad.optimizer.optimizer_prompts import GLOSSARY_TEXT

LOSS_PROMPT = "Below is a response to a task, and the correct answer. Is the response correct? The response does not need to match the correct answer 100%. It is correct as long as they reach the same answer or achieve the same goal. If it's code, also consider its efficiency. If not correct, provide a concise list of errors."

OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve instruction prompt templates or system prompts. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between {new_variable_start_tag} {{improved variable}} {new_variable_end_tag} tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    "System prompt should have the following key elements:\n"
    '- **Role and Purpose**: e.g. "You are a helpful customer support agent. Your goal is to provide clear, accurate, and concise answers to user queries."\n'
    '- **General Behaviour Guidelines**: e.g. "Double-check your calculations and reasoning at each stage to ensure no errors are made."\n'
    '- **Tone and Style of Communication**: e.g. "Use a friendly and professional tone."\n'
    '- **Capabilities**: e.g. "You can provide information on common health issues, explain symptoms, and suggest lifestyle improvements."\n'
    '- **Limitations**: e.g. "You cannot provide legal advice or medical diagnoses. Always direct users to consult a qualified professional."\n'
    '- **Safety or Ethical Guidelines** (if applicable): e.g. "Avoid generating harmful or biased content and ensure privacy is maintained."\n\n'
    "Instruction prompt template should have the following key elements:\n"
    '- **Task Specific Instruction**: e.g. "Translate the following sentence from English to French."\n'
    '- **Input Placeholders**: e.g. "The user has shared the following code snippet for review: {{code_snippet}}."\n'
    '- **Required Output or Structure**: e.g. "Provide your answer in bullet points with a maximum of 5 key points."\n'
    '- **Constraints or Limitations**: e.g. "Do not exceed 150 words in your answer."\n\n'
    f"{GLOSSARY_TEXT}"
)

PERSONAL_FINANCE_ADVISOR_EXAMPLE = (
    "**SYSTEM PROMPT**:\n"
    "You are a personal finance advisor with expertise in budgeting, saving, investing, and debt management. Your goal is to provide sound financial advice based on the user's current situation and goals. You must provide advice that is easy to understand and actionable, avoiding overly technical financial jargon. Ensure that your advice is ethical, realistic, and aligned with the user's best interests.\n\n"
    "**INSTRUCTION PROMPT TEMPLATE**:\n"
    "The user has provided their current financial situation: \n"
    "- Monthly income: {monthly_income}\n"
    "- Monthly expenses: {monthly_expenses}\n"
    "- Savings: {savings}\n"
    "- Debt: {debt}\n"
    "Help them create a budget plan, suggest ways to save, and provide recommendations for managing debt."
)

FITNESS_COACH_EXAMPLE = (
    "**SYSTEM PROMPT**:\n"
    "You are a fitness coach specializing in personalized exercise and nutrition plans. Your goal is to guide users toward their fitness goals by providing tailored workout routines, diet advice, and motivational support. Always consider the user's fitness level, preferences, and any limitations when suggesting exercises or diet plans. Be encouraging, positive, and empathetic in your interactions.\n\n"
    "**INSTRUCTION PROMPT TEMPLATE**:\n"
    "The user has provided the following information: \n"
    "- Weight: {weight}\n"
    "- Height: {height}\n"
    "- Age: {age}\n"
    "- Gender: {gender}\n"
    "- Current fitness level: {fitness_level}\n"
    "- Fitness goals: {fitness_goals}\n"
    "- Workout history: {workout_history}\n"
    "- Nutrition history: {nutrition_history}\n"
    "- Any physical limitations: {limitations}\n"
    "- Any dietary restrictions: {dietary_restrictions}\n"
    "Create a personalized workout plan that includes exercises suitable for their level and aligns with their goals. Include sets, reps, and any tips for proper form. Also provide a nutrition plan."
)

OPTIMIZER_CONSTRAINTS = [
    "If the prompt template contains <input>{input_name}</input>, the new improved prompt template should also contains the same <input>{input_name}</input>.",
    "**Absolute Exclusion of Input Details:** The prompt template should NOT, in any form or manner, contain any information from the inputs, such as parts or the entirety of question, solution, code, etc. Instead, it should have placeholders for the inputs in the form of <input>{input_name}</input>.",
    "System prompts should NOT include any input tags and input placeholders e.g. <input>{input_name}</input> or {input_name}."
]