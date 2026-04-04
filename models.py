"""Pydantic models for LogSentinel environment."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    """A single log entry from a source system."""
    timestamp: str
    source: str
    level: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """What the agent sees at each step."""
    log_entries: List[LogEntry]
    task_description: str
    time_window: str
    remaining_steps: int
    previous_action_result: Optional[str] = None
    incident_context: Optional[Dict[str, Any]] = None


class Action(BaseModel):
    """What the agent can do at each step."""
    action_type: str  # classify_log, detect_incident, assign_severity, recommend_action, correlate_logs, submit_report
    target_log_indices: Optional[List[int]] = None
    classification: Optional[str] = None  # normal, warning, error, critical, security
    severity: Optional[str] = None  # P1, P2, P3, P4
    incident_type: Optional[str] = None  # outage, degradation, security_breach, resource_exhaustion, config_error
    correlated_indices: Optional[List[int]] = None
    recommendation: Optional[str] = None
    report: Optional[Dict[str, Any]] = None


class GroundTruth(BaseModel):
    """Ground truth for grading agent actions."""
    log_classifications: Dict[int, str]  # index -> classification
    incidents: List[Dict[str, Any]]  # list of incidents with correlated indices, type, severity
    expected_severity: Optional[str] = None  # overall severity


class TaskDefinition(BaseModel):
    """Definition of a task in the environment."""
    name: str
    description: str
    difficulty: str
    max_steps: int
    num_logs: int
    num_sources: int
    num_incidents: int
