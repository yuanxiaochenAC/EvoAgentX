from typing import Dict, Any, List, Optional, Tuple
import random
from pydantic import Field

from ...workflow.action_graph import ActionGraph
from ...models.model_configs import LLMConfig
from ...models.base_model import LLMOutputParser
from ...workflow.operators import QAScEnsemble
from .config_pool import load_default_llm_config_pool
from .utils import (
    build_agent_prompt,
    build_judge_prompt,
    default_personas,
    format_transcript,
    collect_last_round_candidates,
    collect_round_candidates,
)
from .pruning import PruningPipeline
from ...agents.customize_agent import CustomizeAgent


class DebateAgentOutput(LLMOutputParser):
    """单个辩手在某一轮的输出结构。"""

    thought: str = Field(default="", description="思考过程")
    argument: str = Field(default="", description="该轮次给出的论据/反驳")
    answer: Optional[str] = Field(default=None, description="该轮次给出的当前答案（可选）")


class DebateJudgeOutput(LLMOutputParser):
    """裁判在辩论结束后的最终判决。"""

    rationale: str = Field(default="", description="评审理由")
    winning_agent_id: int = Field(default=0, description="优胜辩手ID（从0开始）")
    final_answer: str = Field(default="", description="最终答案")


class MultiAgentDebateActionGraph(ActionGraph):
    """多智能体辩论的 ActionGraph 实现（Google MAD 风格）。"""

    name: str = "MultiAgentDebate"
    description: str = "多智能体辩论工作流框架"
    llm_config: LLMConfig = Field(description="用于执行辩论的LLM配置")
    # 可选：外部注入的辩手/裁判 CustomizeAgent（支持不同LLM）
    debater_agents: Optional[List[CustomizeAgent]] = Field(default=None, description="可选：多个辩手CustomizeAgent，执行时将随机选择其一用于该回合/该辩手")
    debater_agent: Optional[CustomizeAgent] = Field(default=None, description="可选：所有辩手复用的单个CustomizeAgent（当未提供 debater_agents 时使用）")
    judge_agent: Optional[CustomizeAgent] = Field(default=None, description="可选：裁判CustomizeAgent，提供则用于裁判阶段")
    # 新增：可选的模型池。当仅给定 agent 数量时，将从该池随机分配给每个 agent（若未提供 debater_agents/agent）
    llm_config_pool: Optional[List[LLMConfig]] = Field(default=None, description="可选：辩手可用的 LLMConfig 候选池，用于仅提供数量时的随机分配")
    # 新增：分组工作流模式（每个席位由一个独立 workflow graph 替代 agent）
    group_graphs_enabled: bool = Field(default=False, description="启用分组图模式：用工作流图替代单个辩手")
    group_graphs: Optional[List[ActionGraph]] = Field(default=None, description="当启用分组图模式时，提供的工作流图列表（长度≥1）")

    # 运行期属性
    _sc_ensemble: Optional[QAScEnsemble] = None

    def init_module(self):
        """初始化模块（创建 LLM，构造可复用运算符）。"""
        super().init_module()
        # 配置冲突与完整性校验
        if self.group_graphs_enabled and (self.debater_agents or self.debater_agent is not None):
            raise ValueError(
                "配置冲突：已启用 group_graphs_enabled 时，不能同时配置 debater_agents 或 debater_agent。"
            )
        if self.debater_agents and self.debater_agent is not None:
            raise ValueError(
                "配置冲突：不能同时提供 debater_agents 与 debater_agent，请择一使用。"
            )
        if self.group_graphs_enabled and (not self.group_graphs or len(self.group_graphs) == 0):
            raise ValueError(
                "配置错误：启用分组图模式时必须提供非空的 group_graphs 列表。"
            )
        if (not self.group_graphs_enabled) and self.group_graphs:
            raise ValueError(
                "配置错误：提供了 group_graphs 但未启用 group_graphs_enabled。请同时启用或移除 group_graphs。"
            )
        self._sc_ensemble = QAScEnsemble(self._llm)
        # 若未配置 llm_config_pool，尝试加载默认池（从环境或本地配置文件）
        if self.llm_config_pool is None:
            try:
                default_pool = load_default_llm_config_pool()
                if default_pool:
                    self.llm_config_pool = default_pool
            except Exception:
                # 静默失败，保持 None
                pass
        # 默认：强制使用 CustomizeAgent（若外部未注入且最终无模型池时创建内置的）
        if not self.debater_agents and self.debater_agent is None and (not self.llm_config_pool):
            self.debater_agent = self._create_default_debater_agent()
        if self.judge_agent is None:
            self.judge_agent = self._create_default_judge_agent()

    def _create_default_debater_agent(self) -> CustomizeAgent:
        """构建默认辩手 CustomizeAgent（XML解析 thought/argument/answer）。"""
        debater_prompt = (
            """
You are debater #{agent_id} (role: {role}). This is round {round_index} of {total_rounds}.

Problem:
{problem}

Conversation so far:
{transcript_text}

Instructions:
- Think briefly (<= 120 words), then present your argument or rebuttal.
- If confident, provide your current answer for this round.
- Your output MUST follow this XML template:

<response>
  <thought>Your brief reasoning</thought>
  <argument>Your argument or rebuttal</argument>
  <answer>Optional current answer; leave empty if uncertain</answer>
</response>
            """
        ).strip()
        inputs = [
            {"name": "problem", "type": "str", "description": "Problem statement"},
            {"name": "transcript_text", "type": "str", "description": "Formatted debate transcript so far"},
            {"name": "role", "type": "str", "description": "Debater role/persona"},
            {"name": "agent_id", "type": "str", "description": "Debater id (string)"},
            {"name": "round_index", "type": "str", "description": "1-based round index"},
            {"name": "total_rounds", "type": "str", "description": "Total rounds"},
        ]
        outputs = [
            {"name": "thought", "type": "str", "description": "Brief reasoning", "required": True},
            {"name": "argument", "type": "str", "description": "Argument or rebuttal", "required": True},
            {"name": "answer", "type": "str", "description": "Optional current answer", "required": False},
        ]
        return CustomizeAgent(
            name="DebaterAgent",
            description="Generate argument/rebuttal and optional answer per debate round.",
            prompt=debater_prompt,
            llm_config=self.llm_config,
            inputs=inputs,
            outputs=outputs,
            parse_mode="xml",
        )

    def _create_default_judge_agent(self) -> CustomizeAgent:
        """构建默认裁判 CustomizeAgent（XML解析 rationale/winning_agent_id/final_answer）。"""
        judge_prompt = (
            """
You are the judge. Based on the multi-round debate, deliver a final decision and answer.

Problem:
{problem}

Debater roles:
{roles_text}

Debate transcript:
{transcript_text}

Return the following XML:
<response>
  <rationale>Your judging rationale</rationale>
  <winning_agent_id>Winning debater ID (integer)</winning_agent_id>
  <final_answer>The final answer</final_answer>
</response>
            """
        ).strip()
        inputs = [
            {"name": "problem", "type": "str", "description": "Problem statement"},
            {"name": "transcript_text", "type": "str", "description": "Formatted debate transcript"},
            {"name": "roles_text", "type": "str", "description": "Roles listing text"},
        ]
        outputs = [
            {"name": "rationale", "type": "str", "description": "Judging rationale", "required": True},
            {"name": "winning_agent_id", "type": "str", "description": "Winning agent id (integer as string)", "required": True},
            {"name": "final_answer", "type": "str", "description": "Final answer", "required": True},
        ]
        return CustomizeAgent(
            name="JudgeAgent",
            description="Deliver final decision and answer based on debate transcript.",
            prompt=judge_prompt,
            llm_config=self.llm_config,
            inputs=inputs,
            outputs=outputs,
            parse_mode="xml",
        )

    def execute(
        self,
        problem: str,
        num_agents: int = 3,
        num_rounds: int = 3,
        judge_mode: str = "llm_judge",  # options: ["llm_judge", "self_consistency"]
        personas: Optional[List[str]] = None,
        return_transcript: bool = True,
        agent_llm_configs: Optional[List[LLMConfig]] = None,
        enable_pruning: bool = False,
        pruning_qp_threshold: float = 0.15,
        pruning_dp_similarity_threshold: float = 0.92,
        pruning_enable_mr: bool = False,
        pruning_mr_llm_config: Optional[LLMConfig] = None,
        pruning_snapshot_mode: bool = False,
        **kwargs,
    ) -> dict:
        """执行辩论工作流（同步）。"""
        state = self._setup_debate(problem, num_agents, num_rounds, personas, agent_llm_configs)
        transcript = self._run_debate_rounds(problem, state)
        # 剪枝（在共识前，对最后一轮候选做筛选和可选纠偏）
        pruning_info = None
        pruning_debug = None
        pruning_rounds_debug: Optional[List[Dict[str, Any]]] = None
        if enable_pruning:
            # 依据参与人数设定最少保留（默认比例 0.3，至少 1 条）
            min_keep = max(1, int(round(state["num_agents"] * 0.3)))
            pipeline = PruningPipeline(
                enable_qp=True,
                enable_dp=True,
                enable_mr=pruning_enable_mr,
                qp_threshold=pruning_qp_threshold,
                dp_similarity_threshold=pruning_dp_similarity_threshold,
                mr_llm_config=pruning_mr_llm_config,
                min_keep_count=min_keep,
            )
            if pruning_snapshot_mode:
                # 为每一轮生成快照（不改变主流程，仅用于展示）
                pruning_rounds_debug = []
                for r in range(state["num_rounds"]):
                    rcands = collect_round_candidates(
                        transcript=transcript, num_agents=state["num_agents"], round_index=r
                    )
                    info_r = pipeline.apply(problem=problem, candidates=rcands)
                    pruning_rounds_debug.append(
                        {
                            "round": r,
                            "before_candidates": rcands,
                            "after_candidates": info_r.get("candidates", []),
                            "mr_suggested": info_r.get("mr_suggested"),
                        }
                    )
            candidates = collect_last_round_candidates(
                transcript=transcript, num_agents=state["num_agents"], last_round_index=state["num_rounds"] - 1
            )
            pruning_info = pipeline.apply(problem=problem, candidates=candidates)
            try:
                pruning_debug = {
                    "before_candidates": candidates,
                    "after_candidates": pruning_info.get("candidates", []),
                    "mr_suggested": pruning_info.get("mr_suggested"),
                }
            except Exception:
                pruning_debug = None
        consensus = self._generate_consensus(problem, state, transcript, judge_mode, pruning_info)
        result: Dict[str, Any] = {
            "final_answer": consensus["final_answer"],
            "winner": consensus.get("winner"),
        }
        if return_transcript:
            result["transcript"] = transcript
        if enable_pruning and pruning_debug is not None:
            result["pruning"] = pruning_debug
        if enable_pruning and pruning_snapshot_mode and pruning_rounds_debug is not None:
            result["pruning_rounds"] = pruning_rounds_debug
        return result

    async def async_execute(
        self,
        problem: str,
        num_agents: int = 3,
        num_rounds: int = 3,
        judge_mode: str = "llm_judge",
        personas: Optional[List[str]] = None,
        return_transcript: bool = True,
        agent_llm_configs: Optional[List[LLMConfig]] = None,
        enable_pruning: bool = False,
        pruning_qp_threshold: float = 0.15,
        pruning_dp_similarity_threshold: float = 0.92,
        pruning_enable_mr: bool = False,
        pruning_mr_llm_config: Optional[LLMConfig] = None,
        pruning_snapshot_mode: bool = False,
        **kwargs,
    ) -> dict:
        """执行辩论工作流（异步）。"""
        state = self._setup_debate(problem, num_agents, num_rounds, personas, agent_llm_configs)
        transcript = await self._run_debate_rounds_async(problem, state)
        pruning_info = None
        pruning_debug = None
        pruning_rounds_debug: Optional[List[Dict[str, Any]]] = None
        if enable_pruning:
            min_keep = max(1, int(round(state["num_agents"] * 0.3)))
            pipeline = PruningPipeline(
                enable_qp=True,
                enable_dp=True,
                enable_mr=pruning_enable_mr,
                qp_threshold=pruning_qp_threshold,
                dp_similarity_threshold=pruning_dp_similarity_threshold,
                mr_llm_config=pruning_mr_llm_config,
                min_keep_count=min_keep,
            )
            if pruning_snapshot_mode:
                pruning_rounds_debug = []
                for r in range(state["num_rounds"]):
                    rcands = collect_round_candidates(
                        transcript=transcript, num_agents=state["num_agents"], round_index=r
                    )
                    info_r = pipeline.apply(problem=problem, candidates=rcands)
                    pruning_rounds_debug.append(
                        {
                            "round": r,
                            "before_candidates": rcands,
                            "after_candidates": info_r.get("candidates", []),
                            "mr_suggested": info_r.get("mr_suggested"),
                        }
                    )
            candidates = collect_last_round_candidates(
                transcript=transcript, num_agents=state["num_agents"], last_round_index=state["num_rounds"] - 1
            )
            pruning_info = pipeline.apply(problem=problem, candidates=candidates)
            try:
                pruning_debug = {
                    "before_candidates": candidates,
                    "after_candidates": pruning_info.get("candidates", []),
                    "mr_suggested": pruning_info.get("mr_suggested"),
                }
            except Exception:
                pruning_debug = None
        consensus = await self._generate_consensus_async(problem, state, transcript, judge_mode, pruning_info)
        result: Dict[str, Any] = {
            "final_answer": consensus["final_answer"],
            "winner": consensus.get("winner"),
        }
        if return_transcript:
            result["transcript"] = transcript
        if enable_pruning and pruning_debug is not None:
            result["pruning"] = pruning_debug
        if enable_pruning and pruning_snapshot_mode and pruning_rounds_debug is not None:
            result["pruning_rounds"] = pruning_rounds_debug
        return result

    def _setup_debate(
        self,
        problem: str,
        num_agents: int,
        num_rounds: int,
        personas: Optional[List[str]],
        agent_llm_configs: Optional[List[LLMConfig]] = None,
    ) -> dict:
        """设置辩论环境。"""
        if num_agents <= 1:
            raise ValueError("num_agents 必须大于 1")
        if num_rounds <= 0:
            raise ValueError("num_rounds 必须为正数")

        roles: List[str] = personas or default_personas(num_agents)
        # 基于用户输入或模型池，准备每个 agent 对象（固定于整个对战周期，不在各轮随机变更）
        agents_for_ids: List[CustomizeAgent] = self._prepare_runtime_debaters(num_agents, agent_llm_configs)
        state: Dict[str, Any] = {
            "problem": problem,
            "num_agents": num_agents,
            "num_rounds": num_rounds,
            "roles": roles,
            "agents": agents_for_ids,
        }
        return state

    def _prepare_runtime_debaters(self, num_agents: int, agent_llm_configs: Optional[List[LLMConfig]]) -> List[CustomizeAgent]:
        """为每个 agent_id 选定在整个辩论中保持不变的 CustomizeAgent。
        优先级：
        1) 用户显式传入 debater_agents → 按长度循环/截断分配给每个位置
        2) 用户传入 debater_agent → 所有位置共用
        3) 传入 agent_llm_configs 或构造时提供 llm_config_pool → 按池随机为每个位置创建默认 debater
        4) 回退到默认的单一 llm_config
        """
        # 若启用分组图模式，不生成内部 Agent（改由 group_graphs 驱动）
        if self.group_graphs_enabled:
            return []
        # 1) 明确的多个 Agent
        if self.debater_agents:
            agents: List[CustomizeAgent] = []
            for i in range(num_agents):
                agents.append(self.debater_agents[i % len(self.debater_agents)])
            return agents
        # 2) 单一 Agent 复用
        if self.debater_agent is not None:
            return [self.debater_agent for _ in range(num_agents)]
        # 3) 依据模型池创建（优先使用调用时传入的 agent_llm_configs）
        if agent_llm_configs and len(agent_llm_configs) > 0:
            return [
                self._create_debater_agent_with_llm(agent_llm_configs[i % len(agent_llm_configs)])
                for i in range(num_agents)
            ]
        pool = self.llm_config_pool
        if pool and len(pool) > 0:
            return [self._create_debater_agent_with_llm(random.choice(pool)) for _ in range(num_agents)]
        # 4) 默认：同一配置
        default_agent = self._create_default_debater_agent()
        return [default_agent for _ in range(num_agents)]

    def _create_debater_agent_with_llm(self, llm_cfg: LLMConfig) -> CustomizeAgent:
        """使用给定 llm 配置创建一个与默认结构一致的辩手 Agent。"""
        debater_prompt = (
            """
You are debater #{agent_id} (role: {role}). This is round {round_index} of {total_rounds}.

Problem:
{problem}

Conversation so far:
{transcript_text}

Instructions:
- Think briefly (<= 120 words), then present your argument or rebuttal.
- If confident, provide your current answer for this round.
- Your output MUST follow this XML template:

<response>
  <thought>Your brief reasoning</thought>
  <argument>Your argument or rebuttal</argument>
  <answer>Optional current answer; leave empty if uncertain</answer>
</response>
            """
        ).strip()
        inputs = [
            {"name": "problem", "type": "str", "description": "Problem statement"},
            {"name": "transcript_text", "type": "str", "description": "Formatted debate transcript so far"},
            {"name": "role", "type": "str", "description": "Debater role/persona"},
            {"name": "agent_id", "type": "str", "description": "Debater id (string)"},
            {"name": "round_index", "type": "str", "description": "1-based round index"},
            {"name": "total_rounds", "type": "str", "description": "Total rounds"},
        ]
        outputs = [
            {"name": "thought", "type": "str", "description": "Brief reasoning", "required": True},
            {"name": "argument", "type": "str", "description": "Argument or rebuttal", "required": True},
            {"name": "answer", "type": "str", "description": "Optional current answer", "required": False},
        ]
        return CustomizeAgent(
            name="DebaterAgent",
            description="Generate argument/rebuttal and optional answer per debate round.",
            prompt=debater_prompt,
            llm_config=llm_cfg,
            inputs=inputs,
            outputs=outputs,
            parse_mode="xml",
        )

    def _run_debate_rounds(self, problem: str, state: dict) -> List[dict]:
        """运行辩论轮次（同步）。返回 transcript。"""
        transcript: List[dict] = []
        num_agents: int = state["num_agents"]
        num_rounds: int = state["num_rounds"]
        roles: List[str] = state["roles"]

        for round_index in range(num_rounds):
            for agent_id in range(num_agents):
                # 分组图模式：调用对应 graph.execute()
                if self.group_graphs_enabled and self.group_graphs:
                    graph = self.group_graphs[agent_id % len(self.group_graphs)]
                    g_inputs = {
                        "problem": problem,
                        "transcript_text": format_transcript(transcript),
                        "role": roles[agent_id],
                        "agent_id": str(agent_id),
                        "round_index": str(round_index + 1),
                        "total_rounds": str(num_rounds),
                    }
                    g_out = graph.execute(**g_inputs)
                    structured = {
                        "argument": g_out.get("argument", g_out.get("output", "")),
                        "answer": g_out.get("answer"),
                        "thought": g_out.get("thought", ""),
                    }
                else:
                    # 若 _setup_debate 已为每个位置确定 Agent，则直接使用
                    selected_agent: Optional[CustomizeAgent] = None
                    agents_for_ids: Optional[List[CustomizeAgent]] = state.get("agents")
                    if agents_for_ids:
                        selected_agent = agents_for_ids[agent_id]
                    elif self.debater_agents:
                        selected_agent = random.choice(self.debater_agents)
                    elif self.debater_agent is not None:
                        selected_agent = self.debater_agent

                    if selected_agent is not None:
                        inputs = {
                            "problem": problem,
                            "transcript_text": format_transcript(transcript),
                            "role": roles[agent_id],
                            "agent_id": str(agent_id),
                            "round_index": str(round_index + 1),
                            "total_rounds": str(num_rounds),
                        }
                        msg = selected_agent(inputs=inputs)
                        structured = msg.content.get_structured_data()
                    else:
                        prompt = build_agent_prompt(
                            problem=problem,
                            transcript_text=format_transcript(transcript),
                            role=roles[agent_id],
                            agent_id=agent_id,
                            round_index=round_index,
                            total_rounds=num_rounds,
                        )
                        response = self._llm.generate(
                            prompt=prompt,
                            parser=DebateAgentOutput,
                            parse_mode="xml",
                        )
                        structured = response.get_structured_data()
                transcript.append(
                    {
                        "agent_id": agent_id,
                        "round": round_index,
                        "role": roles[agent_id],
                        "argument": structured.get("argument", ""),
                        "answer": structured.get("answer"),
                        "thought": structured.get("thought", ""),
                    }
                )
                # 即时打印每个 agent 的产出（同步，完整内容）
                try:
                    arg_full = str(structured.get("argument", "")).strip()
                    ans_full = str(structured.get("answer") or "").strip()
                    print(
                        f"[Round {round_index + 1}] Agent#{agent_id} ({roles[agent_id]})\n"
                        f"Argument: {arg_full}\n"
                        f"Answer: {ans_full}\n"
                    )
                except Exception:
                    pass
        return transcript

    async def _run_debate_rounds_async(self, problem: str, state: dict) -> List[dict]:
        """运行辩论轮次（异步）。返回 transcript。"""
        transcript: List[dict] = []
        num_agents: int = state["num_agents"]
        num_rounds: int = state["num_rounds"]
        roles: List[str] = state["roles"]

        for round_index in range(num_rounds):
            # 每个回合内可并行生成
            transcript_text = format_transcript(transcript)
            # 若分组图模式：逐个调用 graph.execute()（并发复杂性留待后续）
            if self.group_graphs_enabled and self.group_graphs:
                for agent_id in range(num_agents):
                    graph = self.group_graphs[agent_id % len(self.group_graphs)]
                    g_inputs = {
                        "problem": problem,
                        "transcript_text": transcript_text,
                        "role": roles[agent_id],
                        "agent_id": str(agent_id),
                        "round_index": str(round_index + 1),
                        "total_rounds": str(num_rounds),
                    }
                    g_out = graph.execute(**g_inputs)
                    structured = {
                        "argument": g_out.get("argument", g_out.get("output", "")),
                        "answer": g_out.get("answer"),
                        "thought": g_out.get("thought", ""),
                    }
                    transcript.append(
                        {
                            "agent_id": agent_id,
                            "round": round_index,
                            "role": roles[agent_id],
                            "argument": structured.get("argument", ""),
                            "answer": structured.get("answer"),
                            "thought": structured.get("thought", ""),
                        }
                    )
                    # 打印
                    try:
                        print(
                            f"[Round {round_index + 1}] Agent#{agent_id} ({roles[agent_id]})\n"
                            f"Argument: {str(structured.get('argument','')).strip()}\n"
                            f"Answer: {str(structured.get('answer') or '').strip()}\n"
                        )
                    except Exception:
                        pass
            # 若有外部注入的Agent，则并发调用Agent；否则走LLM批量生成
            elif state.get("agents") or self.debater_agents or self.debater_agent is not None:
                import asyncio
                tasks = []
                id_list: List[int] = []
                for agent_id in range(num_agents):
                    agents_for_ids: Optional[List[CustomizeAgent]] = state.get("agents")
                    if agents_for_ids:
                        selected_agent = agents_for_ids[agent_id]
                    elif self.debater_agents:
                        selected_agent = random.choice(self.debater_agents)
                    else:
                        selected_agent = self.debater_agent
                    inputs = {
                        "problem": problem,
                        "transcript_text": transcript_text,
                        "role": roles[agent_id],
                        "agent_id": str(agent_id),
                        "round_index": str(round_index + 1),
                        "total_rounds": str(num_rounds),
                    }
                    tasks.append(selected_agent(inputs=inputs))
                    id_list.append(agent_id)
                messages = await asyncio.gather(*tasks)
                for agent_id, msg in zip(id_list, messages):
                    structured = msg.content.get_structured_data()
                    transcript.append(
                        {
                            "agent_id": agent_id,
                            "round": round_index,
                            "role": roles[agent_id],
                            "argument": structured.get("argument", ""),
                            "answer": structured.get("answer"),
                            "thought": structured.get("thought", ""),
                        }
                    )
                # 即时打印本轮所有 agent 的产出（异步并发完成后逐个打印，完整内容）
                try:
                    for agent_id, msg in zip(id_list, messages):
                        st = msg.content.get_structured_data()
                        arg_full = str(st.get("argument", "")).strip()
                        ans_full = str(st.get("answer") or "").strip()
                        print(
                            f"[Round {round_index + 1}] Agent#{agent_id} ({roles[agent_id]})\n"
                            f"Argument: {arg_full}\n"
                            f"Answer: {ans_full}\n"
                        )
                except Exception:
                    pass
            else:
                prompts: List[Tuple[int, str]] = []
                for agent_id in range(num_agents):
                    prompt = build_agent_prompt(
                        problem=problem,
                        transcript_text=transcript_text,
                        role=roles[agent_id],
                        agent_id=agent_id,
                        round_index=round_index,
                        total_rounds=num_rounds,
                    )
                    prompts.append((agent_id, prompt))

                results = await self._llm.batch_generate_async(
                    batch_messages=[[{"role": "user", "content": p}] for _, p in prompts]
                )
                parsed_list = self._llm.parse_generated_texts(
                    texts=results, parser=DebateAgentOutput, parse_mode="xml"
                )
                for (agent_id, _), parsed in zip(prompts, parsed_list):
                    structured = parsed.get_structured_data()
                    transcript.append(
                        {
                            "agent_id": agent_id,
                            "round": round_index,
                            "role": roles[agent_id],
                            "argument": structured.get("argument", ""),
                            "answer": structured.get("answer"),
                            "thought": structured.get("thought", ""),
                        }
                    )
        return transcript

    def _generate_consensus(
        self, problem: str, state: dict, transcript: List[dict], judge_mode: str, pruning_info: Optional[Dict[str, Any]] = None
    ) -> dict:
        """根据 judge 模式生成最终共识（同步）。"""
        if judge_mode == "self_consistency":
            # 使用 QAScEnsemble 对各辩手最终答案做投票
            agent_final_answers = self._collect_agent_final_answers(state, transcript)
            if len(agent_final_answers) == 0:
                # 无人给出显式答案，则合并最后一轮的论据
                agent_final_answers = [t["argument"] for t in transcript if t.get("argument")]
            sc = self._sc_ensemble.execute(solutions=agent_final_answers)
            return {
                "final_answer": sc["response"],
                "winner": None,
            }

        # 默认 LLM 裁判；若外部注入 judge_agent，则优先使用
        if self.judge_agent is not None:
            roles_text = "\n".join([f"#{i}: {r}" for i, r in enumerate(state["roles"])])
            inputs = {
                "problem": problem,
                "transcript_text": format_transcript(transcript),
                "roles_text": roles_text,
            }
            # 将剪枝摘要附加给裁判（若存在 MR 建议，可以在裁判 prompt 中强调）
            if pruning_info and pruning_info.get("mr_suggested"):
                suggested = pruning_info["mr_suggested"].get("corrected", "")
                if suggested:
                    inputs["problem"] = problem + "\n\n(Consider corrected consolidation, if helpful.)"
            msg = self.judge_agent(inputs=inputs)
            jd = msg.content.get_structured_data()
        else:
            judge_prompt = build_judge_prompt(
                problem=problem,
                transcript_text=format_transcript(transcript),
                roles=state["roles"],
            )
            judge_resp = self._llm.generate(
                prompt=judge_prompt, parser=DebateJudgeOutput, parse_mode="xml"
            )
            jd = judge_resp.get_structured_data()
        return {
            "final_answer": jd.get("final_answer", ""),
            "winner": int(jd.get("winning_agent_id", 0)),
        }

    async def _generate_consensus_async(
        self, problem: str, state: dict, transcript: List[dict], judge_mode: str, pruning_info: Optional[Dict[str, Any]] = None
    ) -> dict:
        """根据 judge 模式生成最终共识（异步）。"""
        if judge_mode == "self_consistency":
            agent_final_answers = self._collect_agent_final_answers(state, transcript)
            if len(agent_final_answers) == 0:
                agent_final_answers = [t["argument"] for t in transcript if t.get("argument")]
            sc = await self._sc_ensemble.async_execute(solutions=agent_final_answers)
            return {
                "final_answer": sc["response"],
                "winner": None,
            }

        if self.judge_agent is not None:
            roles_text = "\n".join([f"#{i}: {r}" for i, r in enumerate(state["roles"])])
            inputs = {
                "problem": problem,
                "transcript_text": format_transcript(transcript),
                "roles_text": roles_text,
            }
            if pruning_info and pruning_info.get("mr_suggested"):
                suggested = pruning_info["mr_suggested"].get("corrected", "")
                if suggested:
                    inputs["problem"] = problem + "\n\n(Consider corrected consolidation, if helpful.)"
            msg = await self.judge_agent(inputs=inputs)
            jd = msg.content.get_structured_data()
        else:
            judge_prompt = build_judge_prompt(
                problem=problem,
                transcript_text=format_transcript(transcript),
                roles=state["roles"],
            )
            judge_resp = await self._llm.async_generate(
                prompt=judge_prompt, parser=DebateJudgeOutput, parse_mode="xml"
            )
            jd = judge_resp.get_structured_data()
        return {
            "final_answer": jd.get("final_answer", ""),
            "winner": int(jd.get("winning_agent_id", 0)),
        }

    def _collect_agent_final_answers(self, state: dict, transcript: List[dict]) -> List[str]:
        """收集每位辩手的最终答案（若有）。"""
        num_agents = state["num_agents"]
        num_rounds = state["num_rounds"]
        final_answers: List[str] = []
        for agent_id in range(num_agents):
            # 查找该辩手在最后一轮的记录
            records = [t for t in transcript if t["agent_id"] == agent_id and t["round"] == num_rounds - 1]
            if len(records) == 0:
                continue
            ans = records[-1].get("answer")
            if ans and isinstance(ans, str) and len(ans.strip()) > 0:
                final_answers.append(ans)
        return final_answers
