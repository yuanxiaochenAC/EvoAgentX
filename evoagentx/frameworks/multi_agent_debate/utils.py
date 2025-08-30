from typing import List, Dict, Any


def default_personas(num_agents: int) -> List[str]:
    """Return default personas in English."""
    base_roles = [
        "Rigorous statistician, emphasizes falsifiability and counterexamples",
        "Pragmatic engineer, focuses on executable plans and edge cases",
        "Devil's advocate, challenges assumptions and proposes alternatives",
        "Knowledge graph expert, stresses structured evidence and provenance",
        "Intuition-driven heuristic scorer, provides quick scoring and heuristics",
    ]
    roles: List[str] = []
    for i in range(num_agents):
        roles.append(base_roles[i % len(base_roles)])
    return roles


def build_agent_prompt(
    problem: str,
    transcript_text: str,
    role: str,
    agent_id: int,
    round_index: int,
    total_rounds: int,
) -> str:
    """Construct agent prompt (XML-structured output)."""
    return f"""
You are debater #{agent_id} (role: {role}). This is round {round_index + 1} of {total_rounds}.

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
""".strip()


def build_judge_prompt(problem: str, transcript_text: str, roles: List[str]) -> str:
    roles_text = "\n".join([f"#{i}: {r}" for i, r in enumerate(roles)])
    return f"""
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
""".strip()


def format_transcript(transcript: List[dict]) -> str:
    if not transcript:
        return "(empty)"
    lines: List[str] = []
    for turn in transcript:
        agent = turn.get("agent_id")
        rd = turn.get("round")
        role = turn.get("role")
        arg = (turn.get("argument") or "").strip()
        ans = (turn.get("answer") or "").strip()
        if ans:
            lines.append(f"[Round {rd}] Agent#{agent} ({role})\nArgument: {arg}\nAnswer: {ans}\n")
        else:
            lines.append(f"[Round {rd}] Agent#{agent} ({role})\nArgument: {arg}\n")
    return "\n".join(lines)


def collect_last_round_candidates(
    transcript: List[dict], num_agents: int, last_round_index: int
) -> List[Dict[str, Any]]:
    """从 transcript 中收集最后一轮每位辩手的候选文本。

    返回形如 [{"agent_id": int, "text": str}] 的列表。
    优先使用该轮的 answer；若没有有效 answer，则回退到 argument。
    """
    candidates: List[Dict[str, Any]] = []
    for agent_id in range(num_agents):
        # 找到该 agent 在最后一轮的记录（若有多条，选最后一条）
        records = [
            t for t in transcript if t.get("agent_id") == agent_id and t.get("round") == last_round_index
        ]
        if not records:
            continue
        rec = records[-1]
        ans = (rec.get("answer") or "").strip()
        arg = (rec.get("argument") or "").strip()
        text = ans if ans else arg
        if text:
            candidates.append({"agent_id": agent_id, "text": text})
    return candidates


def collect_round_candidates(
    transcript: List[dict], num_agents: int, round_index: int
) -> List[Dict[str, Any]]:
    """按指定轮次收集每位辩手的候选文本（优先 answer，回退 argument）。"""
    candidates: List[Dict[str, Any]] = []
    for agent_id in range(num_agents):
        records = [
            t for t in transcript if t.get("agent_id") == agent_id and t.get("round") == round_index
        ]
        if not records:
            continue
        rec = records[-1]
        ans = (rec.get("answer") or "").strip()
        arg = (rec.get("argument") or "").strip()
        text = ans if ans else arg
        if text:
            candidates.append({"agent_id": agent_id, "text": text})
    return candidates

