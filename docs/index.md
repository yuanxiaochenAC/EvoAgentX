# **EvoAgentX**

<p align="center" style="font-size: 1.0rem;">
  <em>An automated framework for evaluating and evolving agentic workflows.</em>
</p>

<p align="center">
  <img src="./assets/framework_en.jpg">
</p>


## What is EvoAgentX
EvoAgentX is an open-source framework for building, evaluating, and evolving LLM-based agents or agentic workflows in an automated, modular, and goal-driven manner.

At its core, EvoAgentX enables developers and researchers to move beyond static prompt chaining or manual workflow orchestration. It introduces a self-evolving agent ecosystem, where AI agents can be constructed, assessed, and optimized through iterative feedback loops—much like how software is continuously tested and improved.

### ✨ Key Features

- 🧱 **Agent Workflow Autoconstruction**
  
  From a single prompt, EvoAgentX builds structured, multi-agent workflows tailored to the task.

- 🔍 **Built-in Evaluation**
  
  It integrates automatic evaluators to score agent behavior using task-specific criteria.

- 🔁 **Self-Evolution Engine**
  
  Agents don’t just work—they learn. EvoAgentX evolves workflows using optimization strategies like retrieval augmentation, mutation, and guided search.

- 🧩 **Plug-and-Play Compatibility**
  
  Easily integrate original [OpenAI](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/models/openai_model.py) and [qwen](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/models/aliyun_model.py) or other popular models, including Claude, Deepseek, kimi models through ([LiteLLM](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/models/litellm_model.py), [siliconflow](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/models/siliconflow_model.py) or [openrouter](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/models/openrouter_model.py)). If you want to use LLMs locally deployed on your own machine, you can try LiteLLM. 

- 🧰 **Comprehensive Built-in Tools**
  
  EvoAgentX ships with a rich set of built-in tools that empower agents to interact with real-world environments.

- 🧠 **Memory Module**
  
  EvoAgentX supports both ephemeral (short-term) and persistent (long-term) memory systems.

- 🧑‍💻 **Human-in-the-Loop (HITL) Interactions**
  
  EvoAgentX supports interactive workflows where humans review, correct, and guide agent behavior.


### 🚀 What You Can Do with EvoAgentX

EvoAgentX isn’t just a framework — it’s your **launchpad for real-world AI agents**.

Whether you're an AI researcher, workflow engineer, or startup team, EvoAgentX helps you **go from a vague idea to a fully functional agentic system** — with minimal engineering and maximum flexibility.

Here’s how:

- 🔍 **Struggling to find the perfect prompt?**  
  EvoAgentX can **automatically explore and evolve prompts** using state-of-the-art self-improving algorithms, all guided by your dataset and goal.

- 🧑‍💻 **Want to supervise the agent and stay in control?**  
  Insert yourself into the loop! EvoAgentX supports **Human-in-the-Loop (HITL)** checkpoints, so you can step in, review, or guide the workflow as needed — and step out again.

- 🧠 **Frustrated by agents that forget everything?**  
  EvoAgentX provides **both short-term and long-term memory modules**, enabling your agents to remember, reflect, and improve across interactions.

- ⚙️ **Lost in manual workflow orchestration?**  
  Just describe your goal — EvoAgentX will **automatically assemble a multi-agent workflow** that matches your intent.

- 🌍 **Want your agents to actually *do* things?**  
  With a rich library of built-in tools (search, code, browser, file I/O, APIs, and more), EvoAgentX empowers agents to **interact with the real world**, not just talk about it.


### 🧰 EvoAgentX Built-in Tools Summary
EvoAgentX ships with a comprehensive suite of **built-in tools**, enabling agents to interact with code environments, search engines, databases, filesystems, images, and browsers. These modular toolkits form the backbone of multi-agent workflows and are easy to extend, customize, and test.

Categories include:
- 🧮 Code Interpreters (Python, Docker)
- 🔍 Search & HTTP Requests (Google, Wikipedia, arXiv, RSS)
- 🗂️ Filesystem Utilities (read/write, shell commands)
- 🧠 Databases (MongoDB, PostgreSQL, FAISS)
- 🖼️ Image Tools (analysis, generation)
- 🌐 Browser Automation (low-level & LLM-driven)

We actively welcome contributions from the community!  
Feel free to propose or submit new tools via [pull requests](https://github.com/EvoAgentX/EvoAgentX/pulls) or [discussions](https://github.com/EvoAgentX/EvoAgentX/discussions).


<details>
<summary>Click to expand full table 🔽</summary>

<br>
  
| Toolkit Name | Description | Code File Path | Test File Path |
|--------------|-------------|----------------|----------------|
| **🧰 Code Interpreters** |  |  |  |
| PythonInterpreterToolkit | Safely execute Python code snippets or local .py scripts with sandboxed imports and controlled filesystem access. | [link](evoagentx/tools/interpreter_python.py) | [link](examples/tools/tools_interpreter.py) |
| DockerInterpreterToolkit | Run code (e.g., Python) inside an isolated Docker container—useful for untrusted code, special deps, or strict isolation. | [link](evoagentx/tools/interpreter_docker.py) | [link](examples/tools/tools_interpreter.py) |
| **🧰 Search & Request Tools** |  |  |  |
| WikipediaSearchToolkit | Search Wikipedia and retrieve results with title, summary, full content, and URL. | [link](evoagentx/tools/search_wiki.py) | [link](examples/tools/tools_search.py) |
| GoogleSearchToolkit | Google Custom Search (official API). Requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID. | [link](evoagentx/tools/search_google.py) | [link](examples/tools/tools_search.py) |
| GoogleFreeSearchToolkit | Google-style search without API credentials (lightweight alternative). | [link](evoagentx/tools/search_google_f.py) | [link](examples/tools/tools_search.py) |
| DDGSSearchToolkit | Search using DDGS with multiple backends and privacy-focused results | [link](evoagentx/tools/search_ddgs.py) | [link](examples/tools/tools_search.py) |
| SerpAPIToolkit | Multi-engine search via SerpAPI (Google/Bing/Baidu/Yahoo/DDG) with optional content scraping. Requires SERPAPI_KEY. | [link](evoagentx/tools/search_serpapi.py) | [link](examples/tools/tools_search.py) |
| SerperAPIToolkit | Google search via SerperAPI with content extraction. Requires SERPERAPI_KEY. | [link](evoagentx/tools/search_serperapi.py) | [link](examples/tools/tools_search.py) |
| RequestToolkit | General HTTP client (GET/POST/PUT/DELETE) with params, form, JSON, headers, raw/processed response, and optional save to file. | [link](evoagentx/tools/request.py) | [link](examples/tools/tools_search.py) |
| ArxivToolkit | Search arXiv for research papers (title, authors, abstract, links/categories). | [link](evoagentx/tools/request_arxiv.py) | [link](examples/tools/tools_search.py) |
| RSSToolkit | Fetch RSS feeds (with optional webpage content extraction) and validate feeds. | [link](evoagentx/tools/rss_feed.py) | [link](examples/tools/tools_search.py) |
| **🧰 FileSystem Tools** |  |  |  |
| StorageToolkit | File I/O utilities: save/read/append/delete, check existence, list files, list supported formats (pluggable storage backends). | [link](evoagentx/tools/storage_file.py) | [link](examples/tools/tools_files.py) |
| CMDToolkit | Execute shell/CLI commands with working directory and timeout control; returns stdout/stderr/return code. | [link](evoagentx/tools/cmd_toolkit.py) | [link](examples/tools/tools_files.py) |
| FileToolkit | File operations toolkit for managing files and directories | [link](evoagentx/tools/file_tool.py) | [link](examples/tools/tools_files.py) |
| **🧰 Database Tools** |  |  |  |
| MongoDBToolkit | MongoDB operations—execute queries/aggregations, find with filter/projection/sort, update, delete, info. | [link](evoagentx/tools/database_mongodb.py) | [link](examples/tools/tools_database.py) |
| PostgreSQLToolkit | PostgreSQL operations—generic SQL execution, targeted SELECT (find), UPDATE, CREATE, DELETE, INFO. | [link](evoagentx/tools/database_postgresql.py) | [link](examples/tools/tools_database.py) |
| FaissToolkit | Vector database (FAISS) for semantic search—insert documents (auto chunk+embed), query by similarity, delete by id/metadata, stats. | [link](evoagentx/tools/database_faiss.py) | [link](examples/tools/tools_database.py) |
| **🧰 Image Handling Tools** |  |  |  |
| ImageAnalysisToolkit | Vision analysis (OpenRouter GPT-4o family): describe images, extract objects/UI info, answer questions about an image. | [link](evoagentx/tools/OpenAI_Image_Generation.py) | [link](examples/tools/tools_images.py) |
| OpenAIImageGenerationToolkit | Text-to-image via OpenAI (DALL·E family) with size/quality/style controls. | [link](evoagentx/tools/OpenAI_Image_Generation.py) | [link](examples/tools/tools_images.py) |
| FluxImageGenerationToolkit | Text-to-image via Flux Kontext Max (BFL) with aspect ratio, seed, format, prompt upsampling, and safety tolerance. | [link](evoagentx/tools/flux_image_generation.py) | [link](examples/tools/tools_images.py) |
| **🧰 Browser Tools** |  |  |  |
| BrowserToolkit | Fine-grained browser automation: initialize, navigate, type, click, resnapshot page, read console logs, and close. | [link](evoagentx/tools/browser_tool.py) | [link](examples/tools/tools_browser.py) |
| BrowserUseToolkit | High-level, natural-language browser automation (navigate, fill forms, click, search, etc.) driven by an LLM. | [link](evoagentx/tools/browser_use.py) | [link](examples/tools/tools_browser.py) |

</details>

## 🔥 EAX Latest News

- **[Aug 2025]** 🚀 **New Survey Released!**  
  Our team just published a comprehensive survey on **Self-Evolving AI Agents**—exploring how agents can learn, adapt, and optimize over time.  
  👉 [Read it on arXiv](https://arxiv.org/abs/2508.07407)

- **[July 2025]** 📚 **EvoAgentX Framework Paper is Live!**  
  We officially published the EvoAgentX framework paper on arXiv, detailing our approach to building evolving agentic workflows.  
  👉 [Check it out](https://arxiv.org/abs/2507.03616)

- **[July 2025]** ⭐️ **1,000 Stars Reached!**  
  Thanks to our amazing community, **EvoAgentX** has surpassed 1,000 GitHub stars!

- **[May 2025]** 🚀 **Official Launch!**  
  **EvoAgentX** is now live! Start building self-evolving AI workflows from day one.  
  🔧 [Get Started on GitHub](https://github.com/EvoAgentX/EvoAgentX)

## 🙋 Support

### Join the Community

📢 Stay connected and be part of the **EvoAgentX** journey!  
🚩 Join our community to get the latest updates, share your ideas, and collaborate with AI enthusiasts worldwide.

- [Discord](https://discord.gg/XWBZUJFwKe) — Chat, discuss, and collaborate in real-time.
- [X (formerly Twitter)](https://x.com/EvoAgentX) — Follow us for news, updates, and insights.
- [WeChat](https://github.com/EvoAgentX/EvoAgentX/blob/main/assets/wechat_info.md) — Connect with our Chinese community.

### Add the meeting to your calendar

📅 Click the link below to add the EvoAgentX Weekly Meeting (Sundays, 16:30–17:30 GMT+8) to your calendar:

👉 [Add to your Google Calendar](https://calendar.google.com/calendar/u/0/r/eventedit?text=EvoAgentX+周会（腾讯会议）&dates=20250629T083000Z/20250629T093000Z&details=会议链接：https://meeting.tencent.com/dm/5UuNxo7Detz0&location=Online&recur=RRULE:FREQ=WEEKLY;BYDAY=SU;UNTIL=20270523T093000Z&ctz=Asia/Shanghai)

👉 [Add to your Tencent Meeting](https://meeting.tencent.com/dm/5UuNxo7Detz0)

👉 [Download the EvoAgentX_Weekly_Meeting.ics file](./EvoAgentX_Weekly_Meeting.ics)

### Contact Information

If you have any questions or feedback about this project, please feel free to contact us. We highly appreciate your suggestions!

- **Email:** evoagentx.ai@gmail.com

We will respond to all questions within 2-3 business days.

### Community Call
- [Bilibili](https://space.bilibili.com/3493105294641286/favlist?fid=3584589186&ftype=create&spm_id_from=333.788.0.0)
- [Youtube](https://studio.youtube.com/playlist/PL_kuPS05qA1hyU6cLX--bJ93Km2-md8AA/edit)
## 🙌 Contributing to EvoAgentX
Thanks go to these awesome contributors

<a href="https://github.com/EvoAgentX/EvoAgentX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=EvoAgentX/EvoAgentX" />
</a>

We appreciate your interest in contributing to our open-source initiative. We provide a document of [contributing guidelines](https://github.com/EvoAgentX/EvoAgentX/blob/main/CONTRIBUTING.md) which outlines the steps for contributing to EvoAgentX. Please refer to this guide to ensure smooth collaboration and successful contributions. 🤝🚀

[![Star History Chart](https://api.star-history.com/svg?repos=EvoAgentX/EvoAgentX&type=Date)](https://www.star-history.com/#EvoAgentX/EvoAgentX&Date)

## 📖 Citation

Please consider citing our work if you find EvoAgentX helpful:

📄 [EvoAgentX](https://arxiv.org/abs/2507.03616)
📄 [Survey Paper](https://arxiv.org/abs/2508.07407)

```bibtex
@article{wang2025evoagentx,
  title={EvoAgentX: An Automated Framework for Evolving Agentic Workflows},
  author={Wang, Yingxu and Liu, Siwei and Fang, Jinyuan and Meng, Zaiqiao},
  journal={arXiv preprint arXiv:2507.03616},
  year={2025}
}
@article{fang202survey,
      title={A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems}, 
      author={Jinyuan Fang and Yanwen Peng and Xi Zhang and Yingxu Wang and Xinhao Yi and Guibin Zhang and Yi Xu and Bin Wu and Siwei Liu and Zihao Li and Zhaochun Ren and Nikos Aletras and Xi Wang and Han Zhou and Zaiqiao Meng},
      year={2025},
      journal={arXiv preprint arXiv:2508.07407},
      url={https://arxiv.org/abs/2508.07407}, 
}
```

