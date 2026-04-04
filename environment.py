"""Core LogSentinel environment with step/reset/state logic."""

import os
import uuid
from typing import Any, Dict, List, Optional

from models import Action, GroundTruth, LogEntry, Observation
from tasks import TASKS
from log_generator import GENERATORS
from graders import grade_action


class LogSentinelEnv:
    """OpenEnv-compliant environment for log analysis and incident classification."""

    def __init__(self):
        self._task_name: str = os.environ.get("MY_ENV_V4_TASK", "log_classification")
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._logs: List[LogEntry] = []
        self._ground_truth: Optional[GroundTruth] = None
        self._max_steps: int = 15
        self._task_description: str = ""
        self._time_window: str = ""
        self._previous_action_result: Optional[str] = None
        self._incident_context: Dict[str, Any] = {}
        self._rewards: List[float] = []
        self._classified_indices: set = set()
        self._detected_incidents: List[Dict[str, Any]] = []

    def reset(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        if task_name:
            self._task_name = task_name

        task_def = TASKS.get(self._task_name)
        if task_def is None:
            self._task_name = "log_classification"
            task_def = TASKS[self._task_name]

        generator = GENERATORS[self._task_name]
        self._logs, self._ground_truth = generator()

        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._max_steps = task_def.max_steps
        self._task_description = task_def.description
        self._time_window = "2024-01-15T10:00:00Z to 2024-01-15T10:35:00Z"
        self._previous_action_result = None
        self._incident_context = {}
        self._rewards = []
        self._classified_indices = set()
        self._detected_incidents = []

        return self._make_observation_response(reward=None, done=False)

    def step(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return observation, reward, done."""
        if self._done:
            return self._make_observation_response(reward=0.0, done=True)

        self._step_count += 1

        # Parse action
        action = Action(**action_data)

        # Grade the action
        reward = grade_action(action, self._ground_truth)
        self._rewards.append(reward)

        # Update state based on action
        self._previous_action_result = self._process_action(action, reward)

        # Check if episode is done
        if action.action_type == "submit_report" or self._step_count >= self._max_steps:
            self._done = True

        return self._make_observation_response(reward=reward, done=self._done)

    def _process_action(self, action: Action, reward: float) -> str:
        """Process action and return feedback string."""
        if action.action_type == "classify_log":
            if action.target_log_indices:
                self._classified_indices.update(action.target_log_indices)
                total = len(self._ground_truth.log_classifications)
                classified = len(self._classified_indices)
                return (
                    f"Classified {len(action.target_log_indices)} log(s) as '{action.classification}'. "
                    f"Reward: {reward:.2f}. Progress: {classified}/{total} logs classified."
                )
            return "No target log indices provided for classification."

        elif action.action_type == "detect_incident":
            self._detected_incidents.append({
                "type": action.incident_type,
                "indices": action.correlated_indices or [],
            })
            self._incident_context["detected_count"] = len(self._detected_incidents)
            return f"Incident detected: type='{action.incident_type}'. Reward: {reward:.2f}."

        elif action.action_type == "assign_severity":
            return f"Severity '{action.severity}' assigned. Reward: {reward:.2f}."

        elif action.action_type == "correlate_logs":
            return f"Correlated {len(action.correlated_indices or [])} logs. Reward: {reward:.2f}."

        elif action.action_type == "recommend_action":
            return f"Recommendation recorded. Reward: {reward:.2f}."

        elif action.action_type == "submit_report":
            return f"Report submitted. Final reward: {reward:.2f}. Episode complete."

        return f"Unknown action type: {action.action_type}. No reward."

    def _make_observation_response(self, reward: Optional[float], done: bool) -> Dict[str, Any]:
        """Build the OpenEnv-compliant response."""
        observation = Observation(
            log_entries=self._logs,
            task_description=self._task_description,
            time_window=self._time_window,
            remaining_steps=max(0, self._max_steps - self._step_count),
            previous_action_result=self._previous_action_result,
            incident_context=self._incident_context if self._incident_context else None,
        )

        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
        }

    @property
    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "task_name": self._task_name,
            "done": self._done,
            "total_reward": sum(self._rewards),
            "rewards": self._rewards,
            "classified_count": len(self._classified_indices),
            "detected_incidents": len(self._detected_incidents),
        }
