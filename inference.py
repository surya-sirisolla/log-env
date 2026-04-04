"""Baseline inference script for LogSentinel environment."""

import json
import os
import sys
import time
import requests
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

TASKS = ["log_classification", "incident_detection", "full_triage"]
BENCHMARK_NAME = "logsentinel"


# ---------------------------------------------------------------------------
# Environment client (talks to the Docker container via HTTP)
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: Optional[float]
    done: bool


class LogSentinelClient:
    """HTTP client for the LogSentinel environment."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    @classmethod
    def from_docker_image(cls, image: str) -> "LogSentinelClient":
        """Start a Docker container and connect to it."""
        import subprocess, socket, time as _time

        # Find free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        # Start container
        container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", f"{port}:7860", image],
            text=True,
        ).strip()

        base_url = f"http://localhost:{port}"

        # Wait for ready
        deadline = _time.time() + 60
        while _time.time() < deadline:
            try:
                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            _time.sleep(1)
        else:
            raise TimeoutError(f"Container {image} did not start in 60s")

        client = cls(base_url)
        client._container_id = container_id
        return client

    def reset(self, task_name: Optional[str] = None) -> StepResult:
        body: Dict[str, Any] = {}
        if task_name:
            body["task_name"] = task_name
        r = self._session.post(f"{self._base}/reset", json=body, timeout=self._timeout)
        r.raise_for_status()
        return self._parse(r.json())

    def step(self, action: Dict[str, Any]) -> StepResult:
        body = {"action": action}
        r = self._session.post(f"{self._base}/step", json=body, timeout=self._timeout)
        r.raise_for_status()
        return self._parse(r.json())

    def close(self):
        cid = getattr(self, "_container_id", None)
        if cid:
            import subprocess
            subprocess.run(["docker", "stop", cid], capture_output=True)
            subprocess.run(["docker", "rm", cid], capture_output=True)

    @staticmethod
    def _parse(payload: Dict[str, Any]) -> StepResult:
        return StepResult(
            observation=payload.get("observation", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a LogSentinel AI agent — an expert SRE/DevOps engineer analyzing application and infrastructure logs.

You will receive a batch of log entries and must take structured actions to analyze them.

Available action types (respond with EXACTLY ONE JSON action per turn):

1. classify_log — Classify one or more logs by severity
   {"action_type": "classify_log", "target_log_indices": [0, 1, 2], "classification": "normal|warning|error|critical|security"}

2. detect_incident — Detect an incident pattern
   {"action_type": "detect_incident", "incident_type": "outage|degradation|security_breach|resource_exhaustion|config_error", "correlated_indices": [3, 5, 7]}

3. assign_severity — Assign priority to a detected incident
   {"action_type": "assign_severity", "severity": "P1|P2|P3|P4", "target_log_indices": [3, 5, 7]}

4. correlate_logs — Group logs that belong to the same incident
   {"action_type": "correlate_logs", "correlated_indices": [3, 5, 7]}

5. recommend_action — Suggest remediation
   {"action_type": "recommend_action", "recommendation": "description of what to do"}

6. submit_report — Submit final structured report (do this last)
   {"action_type": "submit_report", "report": {"incidents": [...], "severity": "P1", "summary": "..."}}

RULES:
- Respond with ONLY a single JSON object — no extra text, no markdown fences.
- Classify ALL logs before detecting incidents.
- After detecting incidents, assign severity and recommend actions.
- Always end with submit_report.
"""


def build_user_message(obs: Dict[str, Any]) -> str:
    """Build a user message from the observation."""
    logs = obs.get("log_entries", [])
    lines = [f"Task: {obs.get('task_description', '')}"]
    lines.append(f"Time window: {obs.get('time_window', '')}")
    lines.append(f"Remaining steps: {obs.get('remaining_steps', 0)}")
    if obs.get("previous_action_result"):
        lines.append(f"Previous result: {obs['previous_action_result']}")
    lines.append("")
    lines.append("LOG ENTRIES:")
    for i, log in enumerate(logs):
        ts = log.get("timestamp", "")
        src = log.get("source", "")
        lvl = log.get("level", "")
        msg = log.get("message", "")
        lines.append(f"[{i}] {ts} {src} {lvl} {msg}")
    return "\n".join(lines)


def call_llm(client: OpenAI, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call the LLM and parse JSON action from response."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def run_task(env: LogSentinelClient, llm: OpenAI, task_name: str) -> List[float]:
    """Run a single task and return list of step rewards."""
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}")

    result = env.reset(task_name=task_name)
    rewards: List[float] = []
    step_num = 0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not result.done:
        step_num += 1
        user_msg = build_user_message(result.observation)
        messages.append({"role": "user", "content": user_msg})

        error_msg = None
        try:
            action = call_llm(llm, messages)
            messages.append({"role": "assistant", "content": json.dumps(action)})
            result = env.step(action)
            reward = result.reward or 0.0
        except Exception as e:
            error_msg = str(e)
            reward = 0.0
            # On error, submit a basic report to end the episode gracefully
            fallback = {"action_type": "submit_report", "report": {"incidents": [], "severity": "P4", "summary": "Error during analysis"}}
            result = env.step(fallback)

        rewards.append(reward)
        done = result.done
        action_str = json.dumps(action, separators=(",", ":")) if error_msg is None else "fallback_report"
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={done} error={error_msg}"
        )

        if done:
            break

    success = sum(rewards) > 0
    score = sum(rewards) / max(len(rewards), 1)
    reward_strs = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step_num} score={score:.3f} rewards={reward_strs}")
    return rewards


def main():
    # Create environment client
    if IMAGE_NAME:
        env = LogSentinelClient.from_docker_image(IMAGE_NAME)
    else:
        # Default: connect to locally running server
        env_url = os.getenv("ENV_URL", "http://localhost:7860")
        env = LogSentinelClient(env_url)

    # Create LLM client
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        all_rewards = {}
        for task in TASKS:
            rewards = run_task(env, llm, task)
            all_rewards[task] = rewards
            print()  # blank line between tasks

        # Summary
        print("=" * 60)
        print("SUMMARY")
        for task, rewards in all_rewards.items():
            total = sum(rewards)
            print(f"  {task}: total_reward={total:.2f} steps={len(rewards)}")
        print("=" * 60)
    finally:
        env.close()


if __name__ == "__main__":
    main()
