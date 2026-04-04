"""Tests for the LogSentinel environment."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from server import app
from environment import LogSentinelEnv
from log_generator import generate_task1_logs, generate_task2_logs, generate_task3_logs
from graders import grade_action
from models import Action, GroundTruth


# ---------------------------------------------------------------------------
# Log generator tests
# ---------------------------------------------------------------------------
class TestLogGenerator:
    def test_task1_generates_10_logs(self):
        logs, gt = generate_task1_logs()
        assert len(logs) == 10
        assert len(gt.log_classifications) == 10

    def test_task1_deterministic(self):
        logs1, gt1 = generate_task1_logs(seed=42)
        logs2, gt2 = generate_task1_logs(seed=42)
        assert [l.message for l in logs1] == [l.message for l in logs2]
        assert gt1.log_classifications == gt2.log_classifications

    def test_task2_generates_20_logs(self):
        logs, gt = generate_task2_logs()
        assert len(logs) == 20
        assert len(gt.incidents) == 2

    def test_task3_generates_30plus_logs(self):
        logs, gt = generate_task3_logs()
        assert len(logs) >= 30
        assert len(gt.incidents) == 3

    def test_task3_has_security_incident(self):
        _, gt = generate_task3_logs()
        types = [inc["type"] for inc in gt.incidents]
        assert "security_breach" in types


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------
class TestGraders:
    def test_correct_classification_rewards(self):
        _, gt = generate_task1_logs()
        # Find an index with a known classification
        idx = 0
        expected_class = gt.log_classifications[idx]
        action = Action(
            action_type="classify_log",
            target_log_indices=[idx],
            classification=expected_class,
        )
        reward = grade_action(action, gt)
        assert reward > 0

    def test_wrong_classification_no_reward(self):
        _, gt = generate_task1_logs()
        idx = 0
        wrong_class = "security" if gt.log_classifications[idx] != "security" else "normal"
        action = Action(
            action_type="classify_log",
            target_log_indices=[idx],
            classification=wrong_class,
        )
        reward = grade_action(action, gt)
        assert reward == 0.0

    def test_incident_detection_reward(self):
        _, gt = generate_task2_logs()
        incident_type = gt.incidents[0]["type"]
        action = Action(
            action_type="detect_incident",
            incident_type=incident_type,
        )
        reward = grade_action(action, gt)
        assert reward > 0

    def test_correlation_reward(self):
        _, gt = generate_task2_logs()
        indices = gt.incidents[0]["correlated_indices"]
        action = Action(
            action_type="correlate_logs",
            correlated_indices=indices,
        )
        reward = grade_action(action, gt)
        assert reward > 0

    def test_report_grading(self):
        _, gt = generate_task2_logs()
        action = Action(
            action_type="submit_report",
            report={
                "incidents": [{"type": "resource_exhaustion"}] * 2,
                "severity": "P2",
                "summary": "Database connection exhaustion causing cascading failures and disk issues",
            },
        )
        reward = grade_action(action, gt)
        assert reward > 0

    def test_unknown_action_zero_reward(self):
        _, gt = generate_task1_logs()
        action = Action(action_type="unknown_action")
        assert grade_action(action, gt) == 0.0


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------
class TestEnvironment:
    def setup_method(self):
        self.env = LogSentinelEnv()

    def test_reset_returns_observation(self):
        result = self.env.reset(task_name="log_classification")
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert result["done"] is False
        assert result["reward"] is None

    def test_reset_has_log_entries(self):
        result = self.env.reset(task_name="log_classification")
        obs = result["observation"]
        assert len(obs["log_entries"]) == 10

    def test_step_returns_reward(self):
        self.env.reset(task_name="log_classification")
        action = {
            "action_type": "classify_log",
            "target_log_indices": [0],
            "classification": "normal",
        }
        result = self.env.step(action)
        assert "reward" in result
        assert isinstance(result["reward"], float)

    def test_submit_report_ends_episode(self):
        self.env.reset(task_name="log_classification")
        action = {
            "action_type": "submit_report",
            "report": {"incidents": [], "severity": "P4", "summary": "No incidents found"},
        }
        result = self.env.step(action)
        assert result["done"] is True

    def test_max_steps_ends_episode(self):
        self.env.reset(task_name="log_classification")
        # Step until max
        for _ in range(20):
            result = self.env.step({"action_type": "classify_log", "target_log_indices": [0], "classification": "normal"})
            if result["done"]:
                break
        assert result["done"] is True

    def test_state_tracks_progress(self):
        self.env.reset(task_name="log_classification")
        state = self.env.state
        assert state["step_count"] == 0
        assert state["episode_id"] is not None

        self.env.step({"action_type": "classify_log", "target_log_indices": [0], "classification": "normal"})
        state = self.env.state
        assert state["step_count"] == 1

    def test_all_tasks_work(self):
        for task in ["log_classification", "incident_detection", "full_triage"]:
            result = self.env.reset(task_name=task)
            assert result["done"] is False
            assert len(result["observation"]["log_entries"]) > 0


# ---------------------------------------------------------------------------
# Server / API tests
# ---------------------------------------------------------------------------
class TestServer:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_tasks(self):
        r = self.client.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()["tasks"]
        assert len(tasks) == 3
        names = {t["name"] for t in tasks}
        assert names == {"log_classification", "incident_detection", "full_triage"}

    def test_reset(self):
        r = self.client.post("/reset", json={})
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert data["done"] is False

    def test_reset_with_task(self):
        r = self.client.post("/reset", json={"task_name": "incident_detection"})
        assert r.status_code == 200
        obs = r.json()["observation"]
        assert len(obs["log_entries"]) == 20

    def test_step(self):
        self.client.post("/reset", json={})
        r = self.client.post("/step", json={
            "action": {
                "action_type": "classify_log",
                "target_log_indices": [0],
                "classification": "normal",
            }
        })
        assert r.status_code == 200
        data = r.json()
        assert "reward" in data
        assert "done" in data

    def test_state(self):
        self.client.post("/reset", json={})
        r = self.client.get("/state")
        assert r.status_code == 200
        state = r.json()
        assert "episode_id" in state
        assert "step_count" in state

    def test_full_episode(self):
        """Test a complete episode flow."""
        self.client.post("/reset", json={"task_name": "log_classification"})

        # Classify some logs
        r = self.client.post("/step", json={
            "action": {
                "action_type": "classify_log",
                "target_log_indices": [0, 1, 2],
                "classification": "normal",
            }
        })
        assert r.status_code == 200
        assert r.json()["done"] is False

        # Submit report
        r = self.client.post("/step", json={
            "action": {
                "action_type": "submit_report",
                "report": {
                    "incidents": [],
                    "severity": "P4",
                    "summary": "Classified logs, no major incidents",
                },
            }
        })
        assert r.status_code == 200
        assert r.json()["done"] is True
