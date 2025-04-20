"""
Required servers: Hirebase, PDF Reader
Links: 
- https://github.com/jhgaylor/hirebase-mcp
- https://github.com/sylphlab/pdf-reader-mcp
"""

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
You are here to review the job opportunities and provide your suggestions.
It is allowed (and expected) to recommend more than 1 actual opportunities for each job category. BUT the jobs must be related to the candidate's background.

## KEY POINTS:
    - You should pick opportunities based on chat history, ensure to include everything.
    - You should COPY AND PASTE the job opportunities content from past tool calls results. 
    - You may provide following information:
        - Job title
        - Company name
        - Location
        - Job description (Complete description from the job posting)
        - Job requirements (Complete requirements from the job posting)
        - Job salary
        - Job posting link
    - Your response must be in a valid JSON format. To reduce confusion, you may try to avoid using "'" and other special characters in your response.

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
        model="gpt-4o-mini",
        openai_key=api_key,
        temperature=0.3,
    )
    
    pdf_file_path = "./examples/test_pdf.pdf"
    config_path = "./examples/mcp.config"
    toolkit = MCPToolkit(config_path=config_path)
    output_file = "./examples/job_recommendations.md"
    
    # Create MCP client and toolkit
    
    
    try:
        # Connect to the MCP server
        await toolkit.connect()
        print(toolkit.get_all_openai_tool_schemas())
        logger.info(f"Connected to MCP server: {toolkit.is_connected()}")
        print(toolkit.get_all_openai_tool_schemas())
        
        # Create ToolCaller agent
        pdf_summarizer_agent = ToolCaller(llm_config=llm_config, max_tool_try = 1)
        pdf_summarizer_agent.add_mcp_toolkit(toolkit)
        
        # Create a user message to process
        logger.info("Creating user message for the ToolCaller agent")
        
        ## ___________ PDF Summarizer Agent ___________
        user_query = f"""
        Assume you are a tech lead and I am seeking for your advice on employment.
        Firstly, you should read and summarize the resume at {pdf_file_path}. You may generate a technical summary, so that if I give the summary to another person, he/she can find a right job for our client.
        Secondly, you should try to retireve real jobs from the internet.
        Thirdly, you should recommend 10 jobs in total from the response of the second step (actual jobs to join a company, with job title, company name, location, job description, job requirements, job posting link, etc). 
        You should ensure you include all information from the second step.
        Lastly, you should also provide a comprehensive summary of your suggestions based on the client's background and past experiences.
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
        job_recommender_agent = ToolCaller(llm_config=llm_config, max_tool_try = 1)
        job_recommender_agent.add_mcp_toolkit(toolkit)
        
        # user_query = f"""
        #     Please recommend 5 actual jobs (actual chance to join a company) to the client 
        #     """
        # input_message = Message(content=user_query, msg_type=MessageType.REQUEST, agent="user")
        
        message_out_searching = await job_recommender_agent.execute(
            action_name=job_recommender_agent.tool_calling_action_name,
            msgs=input_message,
            history=[message_out, message_out_summarizing],
            return_msg_type=MessageType.RESPONSE,
        )
        # print(f"Searching output: {message_out_searching}")
        
        # inter_query = """
        #     \\"job_categories\\": [\\"Engineering Jobs\\"], \\"job_type\\": \\"Full Time\\", \\"location_type\\": \\"In-Person\\", \\"yoe_range\\": {\\"min\\": 7, \\"max\\": 7}, \\"salary_range\\": {\\"min\\": 155000, \\"max\\": 235000, \\"currency\\": \\"USD\\", \\"period\\": \\"year\\"}, \\"date_posted\\": \\"2025-04-18\\", \\"company_link\\": \\"http://bit.ly/shieldai_lever_homepage\\", \\"company_logo\\": \\"https://lever-client-logos.s3.us-west-2.amazonaws.com/e4d7e3f9-4b87-4387-8892-028dfcb1212b-1652726386249.png\\", \\"job_board\\": \\"Lever\\", \\"job_board_link\\": \\"https://jobs.lever.co/shieldai\\", \\"requirements_summary\\": \\"Bachelor\\\\u2019s degree, 7+ years of experience, strong technical background in software engineering\\", \\"visa_sponsored\\": false, \\"company_data\\": {\\"description_summary\\": \\"Bitly offers a suite of tools for link management and analytics to enhance digital connections.\\", \\"linkedin_link\\": \\"https://www.linkedin.com/company/bitly\\", \\"size_range\\": null, \\"industries\\": [\\"Tech, Software & IT Services\\"], \\"subindustries\\": [\\"Digital Media & Entertainment\\"]}, \\"locations\\": [{\\"city\\": \\"Seoul\\", \\"region\\": null, \\"country\\": \\"South Korea\\"}]}", "annotations": null}, {"type": "text", "text": "{\\"_id\\": \\"6801a6d45611cea40b0ab867\\", \\"company_name\\": \\"Shield AI\\", \\"job_title\\": \\"Staff Engineer, Autonomy Software - Field Applications (R3430)\\", \\"description\\": \\"<p>Provide technical expertise and support to customers during the implementation and use of Shield AI enterprise software products. Develop AI & Autonomy applications using the Shield AI enterprise software development kit. Assist the sales team in pre-sales activities and support post-sales deployment and integration of Shield AI enterprise software products.</p><h2>Requirements</h2><ul><li>Bachelor\\\\u2019s degree in Engineering, Computer Science, or a related field.</li><li>7+ years of experience of industry experience or 6+ years of experience plus a master\'s degree.</li><li>3+ years of experience in an integration/applications engineering role.</li><li>2+ years of experience working in a startup environment.</li><li>Strong technical background in software engineering.</li><li>Fluency in written and spoken English</li></ul><h2>Benefits</h2><ul><li>Comprehensive international benefits package</li></ul>\\", \\"application_link\\": \\"https://jobs.lever.co/shieldai/14ca8c8c-5f29-419c-893d-17286e5983c6\\", \\"job_categories\\": [\\"Engineering Jobs\\", \\"Software Engineer Jobs\\"], \\"job_type\\": \\"Full Time\\", \\"location_type\\": \\"In-Person\\", \\"yoe_range\\": {\\"min\\": 7, \\"max\\": 7}, \\"salary_range\\": {\\"min\\": 155000, \\"max\\": 235000, \\"currency\\": \\"USD\\", \\"period\\": \\"year\\"}, \\"date_posted\\": \\"2025-04-18\\", \\"company_link\\": \\"http://bit.ly/shieldai_lever_homepage\\", \\"company_logo\\": \\"https://lever-client-logos.s3.us-west-2.amazonaws.com/e4d7e3f9-4b87-4387-8892-028dfcb1212b-1652726386249.png\\", \\"job_board\\": \\"Lever\\", \\"job_board_link\\": \\"https://jobs.lever.co/shieldai\\", \\"requirements_summary\\": \\"7+ years of industry experience, 3+ years of experience in integration/applications engineering, 2+ years of experience in a startup environment\\", \\"visa_sponsored\\": true, \\"company_data\\": {\\"description_summary\\": \\"Bitly offers a suite of tools for link management and analytics to enhance digital connections.\\", \\"linkedin_link\\": \\"https://www.linkedin.com/company/bitly\\",
        # """
        # inter_message = Message(content=inter_query, msg_type=MessageType.REQUEST, agent="user")
        
        
        final_message_out = await job_recommender_agent.execute(
            action_name=job_recommender_agent.tool_summarizing_action_name,
            msgs=input_message,
            history=[message_out_searching],
            return_msg_type=MessageType.RESPONSE,
            sys_msg=JOB_RECOMMENDER_SUMMARIZING_PROMPT  # Use our custom prompt
        )
        
        # from pdb import set_trace; set_trace()
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
        


    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        raise Exception(f"Traceback: {traceback.format_exc()}")
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