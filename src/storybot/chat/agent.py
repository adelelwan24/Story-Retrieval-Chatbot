"""ReAct agent: the LLM reasons, calls story tools, reads the observations, and answers.

Each step is logged as Thought (the model's text before its tool calls), Action (tool +
arguments) and Observation (the tool's JSON result), so a run can be printed or saved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .prompts import AGENT_SYSTEM
from .tools import StoryTools


@dataclass
class Step:
    thought: str
    tool: str
    arguments: dict
    observation: dict


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stories: list[dict] = field(default_factory=list)   # full stories fetched by id/title, for display

    def trace(self) -> str:
        lines = []
        for i, s in enumerate(self.steps, 1):
            if s.thought:
                lines.append(f"[{i}] {s.thought}")
            lines.append(f"[{i}] Action: {s.tool}({json.dumps(s.arguments, ensure_ascii=False)})")
            obs = json.dumps(s.observation, ensure_ascii=False)
            lines.append(f"[{i}] Observation: {obs[:300]}{'...' if len(obs) > 300 else ''}")
        return "\n".join(lines)


def _clean_answer(text: str) -> str:
    """Drop a leading "Thought: ..." line and an "Answer:" label from the final reply."""
    text = (text or "").strip()
    if text.startswith("Thought:"):
        head, _, rest = text.partition("\n")
        text = rest.strip() or head[len("Thought:"):].strip()
    for label in ("Final Answer:", "Answer:"):
        if text.startswith(label):
            text = text[len(label):].strip()
    return text


class StoryAgent:
    def __init__(self, llm, tools: StoryTools, max_steps: int = 6, max_observation_chars: int = 12000,
                 history_turns: int = 4, system_prompt: str = AGENT_SYSTEM):
        self.llm, self.tools = llm, tools
        self.max_steps, self.max_obs, self.history_turns = max_steps, max_observation_chars, history_turns
        self.system_prompt = system_prompt
        self.history: list[dict] = []    # user / final-answer pairs only, no tool traffic

    def reset(self) -> None:
        self.history.clear()

    def _observation_text(self, obs: dict) -> str:
        s = json.dumps(obs, ensure_ascii=False)
        if len(s) > self.max_obs:
            s = s[:self.max_obs] + ' ... [observation truncated]'
        return s

    def run(self, question: str) -> AgentResult:
        self.tools.shown_stories = []
        messages = [{"role": "system", "content": self.system_prompt},
                    *self.history[-2 * self.history_turns:],
                    {"role": "user", "content": question}]
        steps: list[Step] = []
        seen: dict[str, dict] = {}
        answer = None

        for _ in range(self.max_steps):
            turn = self.llm.chat(messages, tools=self.tools.schemas)
            if not turn.tool_calls:
                answer = turn.content
                break
            messages.append({"role": "assistant", "content": turn.content,
                             "tool_calls": [c.as_message_part() for c in turn.tool_calls]})
            for call in turn.tool_calls:
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                if key in seen:     # same call again: remind instead of re-running
                    obs = {"note": "You already called this tool with these arguments; the result is above. "
                                   "Answer now or try different arguments."}
                else:
                    obs = seen[key] = self.tools.call(call.name, call.arguments)
                steps.append(Step(turn.content, call.name, call.arguments, obs))
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": self._observation_text(obs)})

        if answer is None:   # step budget used up: answer from what was gathered, no more tools
            messages.append({"role": "user", "content": "Stop calling tools. Answer the original question now "
                                                        "from the tool results above."})
            answer = self.llm.chat(messages, tools=None).content

        answer = _clean_answer(answer)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return AgentResult(answer, steps, list(self.tools.shown_stories))

    __call__ = run


def build_agent(retrieval_cfg=None, chat_cfg=None, llm=None, embedder=None, sparse=None, client=None) -> StoryAgent:
    """Everything from the two config files; pass objects to override (tests pass fakes)."""
    from ..config import PROJECT_ROOT, load_config
    from ..retrieval.matching import MetadataMatcher
    from ..retrieval.search import StoryRetriever
    from .llm import build_llm
    from .tools import ToolSettings

    rcfg = retrieval_cfg or load_config()
    ccfg = chat_cfg or load_config(PROJECT_ROOT / "configs" / "chat.yaml")
    t = ccfg.tools
    retriever = StoryRetriever.from_config(rcfg, embedder=embedder, sparse=sparse, client=client,
                                           dense_vector=t.dense_vector, prefetch_limit=t.prefetch_limit)
    matcher = MetadataMatcher(path=rcfg.data.metadata_path, title_threshold=t.title_threshold,
                              genre_threshold=t.genre_threshold)
    llm = llm or build_llm(ccfg.llm)
    tools = StoryTools(retriever, matcher, llm, ToolSettings.from_config(t))
    a = ccfg.agent
    return StoryAgent(llm, tools, max_steps=a.max_steps, max_observation_chars=a.max_observation_chars,
                      history_turns=a.history_turns)
