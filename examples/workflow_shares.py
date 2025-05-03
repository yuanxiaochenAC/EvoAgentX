import os 
from dotenv import load_dotenv 
import asyncio
import sys

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.workflow import WorkFlowGenerator
from evoagentx.tools.mcp import MCPToolkit
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") 

SHARES_GOAL = f"""
Create a comprehensive Chinese market analysis report with the following three sections:

1. INDEX HEALTH CHECK: 
   Analyze the CSI 300 index (HS300) performance over the past year. Include its overall trend, key support/resistance levels, and major turning points. Calculate key metrics such as year-to-date return, volatility, and compare its performance against previous years. Identify whether the index is in bullish, bearish, or consolidation phase.

2. INDUSTRY PERFORMANCE ANALYSIS:
   Retrieve and analyze the performance of major industries in the Chinese market over the past year. For each industry:
   - Calculate the average return over the last 12 months
   - Identify high and low points
   - Measure volatility
   - Compare current performance to historical averages
   Present this as a ranked list from best to worst performing industries.

3. MARKET OUTLOOK SUMMARY:
   Based on the above analysis, provide a concise summary that:
   - Identifies which industries are blooming (showing strong growth and momentum)
   - Which industries are deteriorating (showing weakness or downtrends)
   - How these trends correlate with broader macroeconomic factors
   - Potential opportunities and risks for investors in the coming months

The report should be data-driven, concise, and provide actionable insights for investors considering exposure to different sectors of the Chinese market.
The only initial input for you is this "goal".
"""

AGENT_SUGGESTIONS = f"""
For this complex market analysis task, you'll need to create a multi-agent workflow that efficiently retrieves and processes A-share market data:

1. INDEX ANALYST AGENT:
   - Get CSI 300 constituent stocks using get_hs300_stocks 
   - Retrieve historical data for the index using get_historical_k_data
   - Analyze the performance metrics and identify key trends

2. INDUSTRY CLASSIFIER AGENT:
   - Use get_stock_industry to categorize stocks by industry
   - Group companies into their respective industry sectors
   - Create a mapping of industries for further analysis

3. INDUSTRY PERFORMANCE ANALYZER AGENT:
   - For each industry, retrieve historical data for representative stocks
   - Calculate industry-specific performance metrics
   - Rank industries based on performance criteria
   - Identify outperforming and underperforming sectors

4. SYNTHESIS AGENT:
   - Combine insights from the Index Analyst and Industry Performance agents
   - Create a cohesive narrative about market conditions
   - Generate the final report with all sections properly integrated

The agents should coordinate their analysis, with each focusing on their specific domain while sharing relevant findings with downstream agents. Minimize unnecessary API calls by planning data retrieval strategically - for example, batch similar requests and use sampling for large industries rather than analyzing every stock.
"""


output_file = "workplace/output_shares.md"
mcp_config_path = "workplace/mcp.config"
module_save_path = "workplace/shares_demo_4o_mini.json"

def main(goal=None):

    # LLM configuration
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    # Initialize the language model
    llm = OpenAILLM(config=openai_config)
    
    
    ## Get tools
    mcp_toolkit = MCPToolkit(config_path=mcp_config_path)
    tools = mcp_toolkit.get_tools()
    print(tools)
    
    ## _______________ Workflow Creation _______________
    wf_generator = WorkFlowGenerator(llm=llm, mcp_config_path=mcp_config_path, tools=tools)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal, agent_suggestion=AGENT_SUGGESTIONS)
    # [optional] display workflow
    workflow_graph.display()
    # [optional] save workflow 
    workflow_graph.save_module(module_save_path)
    
    
    ## _______________ Workflow Execution _______________
    #[optional] load saved workflow 
    workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path)

    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    # from pdb import set_trace; set_trace()

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    workflow.init_module()
    output = workflow.execute()
    
    
    ## _______________ Save Output _______________
    try:
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Job recommendations have been saved to {output_file}")
    except Exception as e:
        print(f"Error saving job recommendations: {e}")
    
    # from pdb import set_trace; set_trace()
    print(output)
    
    # verfiy the code
    

if __name__ == "__main__": 
    # Get custom goal from positional argument if provided
    custom_goal = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not custom_goal:
        custom_goal = SHARES_GOAL
    
    # Run the main function with the provided goal
    main(custom_goal)
