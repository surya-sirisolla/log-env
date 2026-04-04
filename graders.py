"""Grading logic for each task in LogSentinel."""

from typing import Any, Dict, List, Set

from models import Action, GroundTruth


def grade_classification(action: Action, ground_truth: GroundTruth) -> float:
    """Grade a classify_log action. Returns reward in [0.0, 1.0]."""
    if action.action_type != "classify_log" or not action.target_log_indices:
        return 0.0

    correct = 0
    total = 0
    for idx in action.target_log_indices:
        if idx in ground_truth.log_classifications:
            total += 1
            expected = ground_truth.log_classifications[idx]
            if action.classification == expected:
                correct += 1

    if total == 0:
        return 0.0

    # Scale reward by classification difficulty
    reward_map = {"normal": 0.1, "warning": 0.15, "error": 0.2, "critical": 0.25, "security": 0.3}
    base_reward = reward_map.get(action.classification or "", 0.1)
    accuracy = correct / total
    return min(1.0, accuracy * base_reward * total)


def grade_incident_detection(action: Action, ground_truth: GroundTruth) -> float:
    """Grade a detect_incident action. Returns reward in [0.0, 1.0]."""
    if action.action_type != "detect_incident":
        return 0.0

    reward = 0.0
    for incident in ground_truth.incidents:
        if action.incident_type == incident["type"]:
            reward += 0.2
            break

    return min(1.0, reward)


def grade_severity(action: Action, ground_truth: GroundTruth) -> float:
    """Grade a severity assignment. Returns reward in [0.0, 1.0]."""
    if action.action_type != "assign_severity" or not action.severity:
        return 0.0

    reward = 0.0
    for incident in ground_truth.incidents:
        if action.severity == incident.get("severity"):
            reward += 0.15
            break
        # Partial credit for being one level off
        severity_order = ["P1", "P2", "P3", "P4"]
        if action.severity in severity_order and incident.get("severity") in severity_order:
            diff = abs(severity_order.index(action.severity) - severity_order.index(incident["severity"]))
            if diff == 1:
                reward += 0.05
                break

    return min(1.0, reward)


def grade_correlation(action: Action, ground_truth: GroundTruth) -> float:
    """Grade log correlation. Returns reward based on overlap with ground truth incidents."""
    if action.action_type != "correlate_logs" or not action.correlated_indices:
        return 0.0

    predicted_set: Set[int] = set(action.correlated_indices)
    best_score = 0.0

    for incident in ground_truth.incidents:
        truth_set: Set[int] = set(incident["correlated_indices"])
        if not truth_set:
            continue

        intersection = predicted_set & truth_set
        precision = len(intersection) / len(predicted_set) if predicted_set else 0
        recall = len(intersection) / len(truth_set) if truth_set else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        best_score = max(best_score, f1 * 0.2)

    return min(1.0, best_score)


def grade_recommendation(action: Action, ground_truth: GroundTruth) -> float:
    """Grade a remediation recommendation. Basic keyword matching."""
    if action.action_type != "recommend_action" or not action.recommendation:
        return 0.0

    rec_lower = action.recommendation.lower()

    # Check for relevant keywords based on incident types
    keyword_groups = {
        "resource_exhaustion": ["connection", "pool", "limit", "scale", "increase", "max_connections", "disk", "space", "cleanup"],
        "outage": ["restart", "failover", "scale", "replicate", "recover", "rollback"],
        "degradation": ["memory", "heap", "gc", "restart", "leak", "limit", "oom"],
        "security_breach": ["block", "firewall", "ip", "waf", "patch", "injection", "sanitize", "rate limit"],
        "config_error": ["config", "setting", "revert", "correct", "update"],
    }

    reward = 0.0
    for incident in ground_truth.incidents:
        itype = incident["type"]
        keywords = keyword_groups.get(itype, [])
        matches = sum(1 for kw in keywords if kw in rec_lower)
        if matches > 0:
            reward += min(0.15, matches * 0.03)

    return min(1.0, reward)


def grade_report(action: Action, ground_truth: GroundTruth) -> float:
    """Grade the final incident report. Returns reward in [0.0, 0.3]."""
    if action.action_type != "submit_report" or not action.report:
        return 0.0

    report = action.report
    reward = 0.0

    # Check for required report fields
    required_fields = ["incidents", "severity", "summary"]
    for field in required_fields:
        if field in report:
            reward += 0.05

    # Check incident count matches
    if "incidents" in report:
        incidents = report["incidents"]
        if isinstance(incidents, list):
            expected_count = len(ground_truth.incidents)
            actual_count = len(incidents)
            if actual_count == expected_count:
                reward += 0.1
            elif abs(actual_count - expected_count) == 1:
                reward += 0.05

    # Check if summary is non-trivial
    if "summary" in report and isinstance(report["summary"], str) and len(report["summary"]) > 20:
        reward += 0.05

    return min(0.3, reward)


GRADER_MAP = {
    "classify_log": grade_classification,
    "detect_incident": grade_incident_detection,
    "assign_severity": grade_severity,
    "correlate_logs": grade_correlation,
    "recommend_action": grade_recommendation,
    "submit_report": grade_report,
}


def grade_action(action: Action, ground_truth: GroundTruth) -> float:
    """Grade any action against ground truth. Returns reward in [0.0, 1.0]."""
    grader = GRADER_MAP.get(action.action_type)
    if grader is None:
        return 0.0
    return grader(action, ground_truth)
