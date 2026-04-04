"""Task definitions for LogSentinel environment."""

from models import TaskDefinition

TASKS = {
    "log_classification": TaskDefinition(
        name="log_classification",
        description=(
            "Classify 10 log entries by severity and type from a single source. "
            "Each log should be classified as: normal, warning, error, critical, or security. "
            "Submit a final report when done."
        ),
        difficulty="easy",
        max_steps=15,
        num_logs=10,
        num_sources=1,
        num_incidents=0,
    ),
    "incident_detection": TaskDefinition(
        name="incident_detection",
        description=(
            "Analyze 20 log entries from 3 different sources. Classify each log, "
            "detect 2 incidents by correlating related logs across sources, "
            "and assign correct severity (P1-P4) to each incident. "
            "Submit a structured incident report when done."
        ),
        difficulty="medium",
        max_steps=25,
        num_logs=20,
        num_sources=3,
        num_incidents=2,
    ),
    "full_triage": TaskDefinition(
        name="full_triage",
        description=(
            "Full incident triage with 30+ log entries from 5 sources including noise. "
            "Classify all logs, detect 3 incidents (one is a subtle security issue), "
            "correlate logs across sources, assign severity, recommend remediation "
            "for each incident, and submit a comprehensive incident report. "
            "Includes red herrings and ambiguous logs."
        ),
        difficulty="hard",
        max_steps=35,
        num_logs=34,
        num_sources=5,
        num_incidents=3,
    ),
}

TASK_LIST = list(TASKS.keys())
