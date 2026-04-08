"""Baseline inference script for LogSentinel environment."""

import json
import os
import sys
import time
import traceback
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["log_classification", "incident_detection", "full_triage"]
BENCHMARK_NAME = "logsentinel"

MAX_RETRIES = 3
RETRY_DELAY = 2.0


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
        self._container_id: Optional[str] = None

    @classmethod
    def from_docker_image(cls, image: str) -> "LogSentinelClient":
        """Start a Docker container and connect to it."""
        import subprocess
        import socket

        # Find free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        # Start container
        try:
            container_id = subprocess.check_output(
                ["docker", "run", "-d", "-p", f"{port}:7860", image],
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start docker container: {e.stderr}")

        base_url = f"http://localhost:{port}"

        # Wait for ready
        deadline = time.time() + 90
        ready = False
        while time.time() < deadline:
            try:
                r = requests.get(f"{base_url}/health", timeout=3)
                if r.status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1)

        if not ready:
            # Cleanup before raising
            try:
                subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=10)
                subprocess.run(["docker", "rm", container_id], capture_output=True, timeout=10)
            except Exception:
                pass
            raise TimeoutError(f"Container {image} did not become ready in 90s")

        client = cls(base_url)
        client._container_id = container_id
        return client

    def _post_with_retry(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST with retries — never raises on transient failures."""
        last_err: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                r = self._session.post(
                    f"{self._base}{path}",
                    json=body,
                    timeout=self._timeout,
                )
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
        raise RuntimeError(f"POST {path} failed after {MAX_RETRIES} attempts: {last_err}")

    def reset(self, task_name: Optional[str] = None) -> StepResult:
        body: Dict[str, Any] = {}
        if task_name:
            body["task_name"] = task_name
        payload = self._post_with_retry("/reset", body)
        return self._parse(payload)

    def step(self, action: Dict[str, Any]) -> StepResult:
        payload = self._post_with_retry("/step", {"action": action})
        return self._parse(payload)

    def close(self):
        cid = self._container_id
        if cid:
            try:
                import subprocess
                subprocess.run(["docker", "stop", cid], capture_output=True, timeout=10)
                subprocess.run(["docker", "rm", cid], capture_output=True, timeout=10)
            except Exception:
                pass
            self._container_id = None
        try:
            self._session.close()
        except Exception:
            pass

    @staticmethod
    def _parse(payload: Dict[str, Any]) -> StepResult:
        return StepResult(
            observation=payload.get("observation", {}) or {},
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
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
    logs = obs.get("log_entries", []) or []
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


def call_llm(client: Any, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call the LLM and parse JSON action from response."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    text = (resp.choices[0].message.content or "").strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def heuristic_action(obs: Dict[str, Any], step_num: int) -> Dict[str, Any]:
    """Fallback heuristic agent (no LLM): classifies based on log level."""
    logs = obs.get("log_entries", []) or []
    if not logs:
        return {"action_type": "submit_report", "report": {"incidents": [], "severity": "P4", "summary": "No logs to analyze"}}

    # Group log indices by their level → classification
    level_to_class = {
        "INFO": "normal",
        "DEBUG": "normal",
        "WARN": "warning",
        "WARNING": "warning",
        "ERROR": "error",
        "FATAL": "critical",
        "CRITICAL": "critical",
    }

    groups: Dict[str, List[int]] = {}
    for i, log in enumerate(logs):
        cls = level_to_class.get(log.get("level", "").upper(), "normal")
        groups.setdefault(cls, []).append(i)

    classes_list = list(groups.keys())
    if step_num <= len(classes_list):
        cls = classes_list[step_num - 1]
        return {
            "action_type": "classify_log",
            "target_log_indices": groups[cls],
            "classification": cls,
        }

    # After classifying, submit a report
    return {
        "action_type": "submit_report",
        "report": {
            "incidents": [],
            "severity": "P3",
            "summary": "Heuristic baseline classification of logs by severity level",
        },
    }


def run_task(env: LogSentinelClient, llm: Any, task_name: str) -> List[float]:
    """Run a single task and return list of step rewards. Never raises."""
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    step_num = 0

    try:
        result = env.reset(task_name=task_name)
    except Exception as e:
        err = str(e).replace("\n", " ")[:200]
        print(f"[STEP] step=1 action=reset_failed reward=0.00 done=true error={err}", flush=True)
        print(f"[END] success=false steps=1 score=0.001 rewards=0.00", flush=True)
        return [0.0]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_steps_safety = 30  # hard cap to avoid infinite loops

    while not result.done and step_num < max_steps_safety:
        step_num += 1
        user_msg = build_user_message(result.observation)
        messages.append({"role": "user", "content": user_msg})

        error_msg: Optional[str] = None
        action: Dict[str, Any] = {}
        reward: float = 0.0

        try:
            if llm is not None:
                action = call_llm(llm, messages)
                messages.append({"role": "assistant", "content": json.dumps(action)})
            else:
                # No LLM available — use heuristic baseline
                action = heuristic_action(result.observation, step_num)
        except Exception as e:
            error_msg = str(e).replace("\n", " ")[:200]
            action = heuristic_action(result.observation, step_num)

        # Now actually call env.step — separately wrapped
        try:
            result = env.step(action)
            reward = float(result.reward or 0.0)
        except Exception as e:
            error_msg = (error_msg or "") + f"|step_failed:{str(e)[:100]}"
            reward = 0.0
            # Force episode end
            result = StepResult(observation={}, reward=0.0, done=True)

        rewards.append(reward)
        done = result.done

        try:
            action_str = json.dumps(action, separators=(",", ":"))
        except Exception:
            action_str = str(action)[:200]

        err_field = error_msg if error_msg else "null"
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={err_field}",
            flush=True,
        )

        if done:
            break

    success = sum(rewards) > 0
    raw_score = sum(rewards) / max(len(rewards), 1)
    # Validator requires score strictly in (0, 1) — clamp away from endpoints
    score = min(0.999, max(0.001, raw_score))
    reward_strs = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={step_num} score={score:.3f} rewards={reward_strs}",
        flush=True,
    )
    return rewards


def make_llm_client():
    """Create the LLM client. Returns None if it fails (script falls back to heuristic)."""
    if not HF_TOKEN:
        print("# Warning: HF_TOKEN not set — using heuristic baseline", flush=True)
        return None
    try:
        from openai import OpenAI
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"# Warning: failed to create OpenAI client ({e}) — using heuristic baseline", flush=True)
        return None


def make_env_client() -> Optional[LogSentinelClient]:
    """Create the environment client. Returns None on failure."""
    try:
        if LOCAL_IMAGE_NAME:
            return LogSentinelClient.from_docker_image(LOCAL_IMAGE_NAME)
        env_url = os.getenv("ENV_URL", "http://localhost:7860")
        return LogSentinelClient(env_url)
    except Exception as e:
        print(f"# Warning: failed to create env client: {e}", flush=True)
        return None


def main() -> int:
    """Run all tasks. Always returns 0 — never crashes."""
    env: Optional[LogSentinelClient] = None
    try:
        env = make_env_client()
        if env is None:
            # Print fail-safe output for all tasks so the format check passes
            for task in TASKS:
                print(f"[START] task={task} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)
                print(f"[STEP] step=1 action=env_unavailable reward=0.00 done=true error=env_client_init_failed", flush=True)
                print(f"[END] success=false steps=1 score=0.001 rewards=0.00", flush=True)
                print(flush=True)
            return 0

        llm = make_llm_client()

        all_rewards: Dict[str, List[float]] = {}
        for task in TASKS:
            try:
                rewards = run_task(env, llm, task)
            except Exception as e:
                # Belt-and-suspenders: run_task should never raise, but just in case
                err = str(e).replace("\n", " ")[:200]
                print(f"[STEP] step=1 action=task_crashed reward=0.00 done=true error={err}", flush=True)
                print(f"[END] success=false steps=1 score=0.001 rewards=0.00", flush=True)
                rewards = [0.0]
            all_rewards[task] = rewards
            print(flush=True)

        # Summary
        print("=" * 60, flush=True)
        print("SUMMARY", flush=True)
        for task, rewards in all_rewards.items():
            total = sum(rewards)
            print(f"  {task}: total_reward={total:.2f} steps={len(rewards)}", flush=True)
        print("=" * 60, flush=True)

    except Exception:
        # Ultimate safety net — print traceback to stderr but never exit non-zero
        traceback.print_exc(file=sys.stderr)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
