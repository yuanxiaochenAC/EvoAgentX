🌟 Wonderful Workflow Corpus
============================

This repository is a collection of reusable workflows, each defined by a `workflow.json` (workflow logic) and a `tools.json` (tool definitions). All workflows can be executed through the universal script `execute_workflow.py` by passing in a goal string written in natural language.

The corpus includes both simple workflows, such as **Arxiv Daily Digest**, **Recipe Generator**, or **Tetris Game**, which can be run directly with a short goal, and advanced workflows like **Invest (Stock Analysis)**, which combine file discovery, data loading, and multi-stage analysis. This way the corpus supports everything from quick experiments to more complex analysis tasks.  

------------------------------------------------------------

🚀 Quick Start
--------------

1. Install dependencies:
   pip install -r requirements.txt

2. Make sure your OpenAI API Key is set either as an environment variable or in a .env file:
   OPENAI_API_KEY=sk-xxxxxx

3. Execute a workflow (general command):
   ```bash
   python execute_workflow.py `
     --workflow <path-to-workflow.json> `
     --goal "<your-goal>" `
     --output <result-file>
   ```

------------------------------------------------------------

📂 Example Workflows
--------------------

1. Arxiv Daily Digest
   Recommend the latest academic papers from arXiv based on keywords.

   ```bash
   python Wonderful_workflow_corpus/execute_workflow.py `
     --workflow Wonderful_workflow_corpus/arxiv_daily_digest/workflow.json `
     --goal "Please recommend the latest papers on multi-agent systems in natural language processing. Summarize each paper in 3 sentences and provide direct arXiv links." `
     --output arxiv_digest.md
   ```

------------------------------------------------------------

2. Recipe Generator
   Generate a recipe based on user preferences.

   ```bash
   python Wonderful_workflow_corpus/execute_workflow.py `
     --workflow Wonderful_workflow_corpus/recipe_generator/workflow.json `
     --goal "Generate a healthy vegetarian dinner recipe that uses tofu, broccoli, and garlic. Include step-by-step instructions and estimated cooking time." `
     --output recipe.md
   ```

------------------------------------------------------------

3. Tetris Game Generator
   Generate HTML/CSS/JavaScript code for a browser-based Tetris game.

   ```bash
   python Wonderful_workflow_corpus/execute_workflow.py `
     --workflow Wonderful_workflow_corpus/tetris_game/workflow.json `
     --goal "Generate a playable Tetris game with scoring, level progression, and keyboard controls. The game should run in any modern browser." `
     --output tetris.html
   ```

   Open tetris.html in a browser to play.

------------------------------------------------------------

📊 Advanced Workflow
--------------------

4. Example: Stock Analysis
Run the full pipeline with:

```bash
python Wonderful_workflow_corpus/invest/stock_analysis.py
```

The process will:

1. Create required folders.
2. Fetch stock data (if not already available).
3. Generate graphs (if not already available).
4. Execute the stock analysis workflow.
5. Save a Markdown report and an HTML report.

------------------------------------------------------------

⚠️ Notes
- `stock_analysis.py` automatically selects the workflow file based on the operating system: `workflow_windows.json` for Windows users, and `workflow.json` for macOS/Linux users.
- Windows users: workflows use `cmd /c dir /a /b /s "<PATH>"` for file discovery.
- macOS/Linux users: workflows use `find "<PATH>" -type f -print` for file discovery.
- If cross-platform support is needed, ensure workflow prompts strictly differentiate between OS commands.

------------------------------------------------------------

✨ Extending the Corpus
-----------------------

You can add new workflows by:
1. Writing a workflow.json and tools.json in a new subdirectory.
2. Running them with the same executor:
   python execute_workflow.py --workflow path/to/new/workflow.json --goal "<goal>" --output result.md

------------------------------------------------------------