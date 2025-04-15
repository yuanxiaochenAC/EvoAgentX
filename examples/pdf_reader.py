import asyncio
import os
import json
import http.server
import socketserver
import threading
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from evoagentx.agents.tool_caller import ToolCaller
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.core.logging import logger
from evoagentx.core.message import Message, MessageType
from evoagentx.prompts.tool_caller import TOOL_CALLER_PROMPT
load_dotenv()

# =================== PROMPTS ===================
# PDF RESUME SUMMARIZER - SUMMARIZING PROMPT
PDF_RESUME_SUMMARIZING_PROMPT = """
You are a senior technical recruiter with expertise in career guidance for technology professionals.

Your task is to provide a comprehensive professional assessment of the candidate based on the resume data gathered.
Focus on delivering actionable insights that would help another recruiter understand this candidate's profile and potential.

In your assessment, you MUST include:

1. COMPREHENSIVE TECHNICAL PROFILE:
   - Provide a detailed breakdown of the candidate's technical background, skills, and expertise
   - Analyze programming languages, frameworks, and technologies they've worked with
   - Evaluate their level of expertise in each area (beginner, intermediate, expert)
   - Assess their experience with different development environments, platforms, and methodologies

2. STRENGTHS ANALYSIS:
   - Identify 5-7 specific technical and professional strengths
   - Provide CONCRETE EXAMPLES from their resume that demonstrate each strength
   - Evaluate how these strengths align with current industry demands
   - Assess which strengths would be most valuable to potential employers

3. AREAS FOR DEVELOPMENT:
   - Identify 3-5 skill gaps or areas where further development would benefit their career
   - Suggest specific ways they could address these gaps (certifications, training, projects)
   - Explain how addressing these areas would improve their employability
   - Consider both technical and soft skills in your assessment

4. CAREER TRAJECTORY:
   - Analyze their career progression and patterns
   - Assess whether their experience demonstrates growth, specialization, or versatility
   - Evaluate how well-positioned they are for senior/leadership roles if applicable
   - Identify potential next career moves based on their background


### Response Format
You MUST respond with a valid JSON object in the following format. Do not include any text before or after the JSON object:
```json
{
    "summary": "Your comprehensive assessment goes here..."
}
```
All your response should be put into the `summary` field as a long string.
"""

# JOB RECOMMENDER - SUMMARIZING PROMPT
JOB_RECOMMENDER_SUMMARIZING_PROMPT = """
You are a world-class technical career consultant who provides elite-level career guidance to technology professionals.
Your job is to select / filtering existing job opportunities based on your own judgement, considering the client's background and preferences.
You will give comment on each job opportunity, and then provide short summary of your suggestions based on the client's background and past experiences.
The job opportunities means real opporunity to join a company instead of type of job titles.

## Contents:
### Job Opportunities:
    - You should pick opportunities based on chat history, ensure to include everything.
    - You should copy and paste the job opportunities from past tool calls. Information should stay changed but you may do some reformating.
    - You may provide following information:
        - Job title
        - Company name
        - Location
        - Job description
        - Job requirements
        - Job salary
        - Job posting link

### Your suggestions:
    - You should rank the job opportunities based on the client's background and past experiences.
    - You may provide 3-5 suggestions.
    

### Response Format
You MUST respond with a valid JSON object in the following format. Do not include any text before or after the JSON object:
```json
{
    "summary": "Your comprehensive job recommendations go here..."
}
```
All your response should be put into the `summary` field as a long string. The string should be able to form a .md markdown file.
"""

async def main():
    logger.info("=== ToolCaller Agent with MCP Tools Example ===")
    
    # Load environment variables
    
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize the language model
    llm_config = OpenAILLMConfig(
        llm_type="OpenAILLM",
        model="gpt-4o-mini",  # Using GPT-4o-mini for tool usage capabilities
        openai_key=api_key,
        temperature=0.7,
        max_tokens=3000,
    )
    
    pdf_file_path = "./tests/test_pdf.pdf"
    config_path = "tests/mcp.config"
    toolkit = MCPToolkit(config_path=config_path)
    output_file = "job_recommendations.md"
    
    # Create MCP client and toolkit
    
    
    try:
        # Connect to the MCP server
        await toolkit.connect()
        print(toolkit.get_all_openai_tool_schemas())
        logger.info(f"Connected to MCP server: {toolkit.is_connected()}")
        print(toolkit.get_all_openai_tool_schemas())
        
        # Create ToolCaller agent
        pdf_summarizer_agent = ToolCaller(llm_config=llm_config)
        pdf_summarizer_agent.add_mcp_toolkit(toolkit)
        
        # Create a user message to process
        logger.info("Creating user message for the ToolCaller agent")
        
        ## ___________ PDF Summarizer Agent ___________
        user_query = f"""
        Assume you are a tech lead and I am seeking for your advice on employment.
        Please read and summarize my resume at {pdf_file_path}, interms of pros, cons, and suggestions for titled jobs.
        It should be a very technical summary, so that if I give the summary to another person, he/she can find a right job for me.
        If you already have the summary, please find way to find real jobs and then recommend me 5 jobs from these jobs (actual jobs to join a company, with job title, company name, location, job description, job requirements, job salary, job posting link) for our friend based on the summary.
        """
        input_message = Message(content=user_query, msg_type=MessageType.REQUEST, agent="user")
        
        message_out = await pdf_summarizer_agent.execute(
            action_name=pdf_summarizer_agent.tool_calling_action_name,
            msgs=input_message,
            return_msg_type=MessageType.RESPONSE,
        )
        # print(f"Searching output: {message_out}")
        
        message_out_summarizing = await pdf_summarizer_agent.execute(
            action_name=pdf_summarizer_agent.tool_summarizing_action_name,
            msgs=input_message,
            history=[message_out],
            return_msg_type=MessageType.RESPONSE,
            sys_msg=PDF_RESUME_SUMMARIZING_PROMPT  # Use our custom prompt
        )
        # print(f"Summarizing output: {message_out_summarizing}")
        
        
        ## ___________ Job Recommender Agent ___________
        job_recommender_agent = ToolCaller(llm_config=llm_config)
        job_recommender_agent.add_mcp_toolkit(toolkit)
        
        # user_query = f"""
        #     Please recommend me 5 jobs (actual jobs to join a company) for our friend based on the following information: {message_out_summarizing.content}
        #     """
        input_message = Message(content=user_query, msg_type=MessageType.REQUEST, agent="user")
        
        message_out_searching = await job_recommender_agent.execute(
            action_name=job_recommender_agent.tool_calling_action_name,
            msgs=input_message,
            history=[message_out, message_out_summarizing],
            return_msg_type=MessageType.RESPONSE,
        )
        # print(f"Searching output: {message_out_searching}")
        
        final_message_out = await job_recommender_agent.execute(
            action_name=job_recommender_agent.tool_summarizing_action_name,
            msgs=input_message,
            history=[message_out, message_out_summarizing, message_out_searching],
            return_msg_type=MessageType.RESPONSE,
            sys_msg=JOB_RECOMMENDER_SUMMARIZING_PROMPT  # Use our custom prompt
        )
        
        # print(f"Summarizing output: {final_message_out}")
        
        # Extract the summary from the final message output
        try:
            summary_content = json.loads(str(final_message_out.content))["summary"]
            
            # Write to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary_content)
            
            logger.info(f"Successfully saved job recommendations to {output_file}")
            print(f"Job recommendations have been saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving job recommendations: {e}")
            print(f"Error saving job recommendations: {e}")
        
        # from pdb import set_trace; set_trace()


    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Clean up resources
        print("Disconnecting from MCP server")
        try:
            await toolkit.disconnect()
        except asyncio.CancelledError as e:
            # Safely handle the cancellation error
            print(f"Caught cancellation during disconnect: {e}")
            print("This is expected behavior with the MCP client and can be safely ignored.")
        except Exception as e:
            print(f"Error during disconnect: {e}")
        logger.info("Disconnected from MCP server")
    
    logger.info("\nExample completed!")

if __name__ == "__main__":
    asyncio.run(main())