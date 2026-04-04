"""Synthetic log generation engine with ground truth for grading."""

import random
from typing import Any, Dict, List, Tuple

from models import GroundTruth, LogEntry


def generate_task1_logs(seed: int = 42) -> Tuple[List[LogEntry], GroundTruth]:
    """Generate 10 straightforward logs from a single source for classification."""
    rng = random.Random(seed)

    templates = [
        ("INFO", "Health check passed: status=200 latency=12ms", "normal"),
        ("ERROR", "Connection refused: postgres://db-primary:5432 - ECONNREFUSED", "error"),
        ("WARN", "Memory usage at 78% - approaching threshold", "warning"),
        ("FATAL", "OutOfMemoryError: Java heap space - process killed by OOM killer", "critical"),
        ("INFO", "Request processed successfully: GET /api/users 200 45ms", "normal"),
        ("ERROR", "Unauthorized access attempt from IP 203.0.113.42 - invalid token with suspicious payload", "security"),
        ("WARN", "Disk usage on /var/log at 85% - rotation recommended", "warning"),
        ("ERROR", "Database query timeout after 30000ms: SELECT * FROM orders WHERE status='pending'", "error"),
        ("INFO", "Scheduled backup completed: 2.3GB written to s3://backups/daily/", "normal"),
        ("FATAL", "Kernel panic - not syncing: Fatal exception in interrupt", "critical"),
    ]

    rng.shuffle(templates)
    logs = []
    classifications = {}

    for i, (level, message, classification) in enumerate(templates):
        log = LogEntry(
            timestamp=f"2024-01-15T10:{i:02d}:{rng.randint(0, 59):02d}Z",
            source="app-server-1",
            level=level,
            message=message,
            metadata={"request_id": f"req-{rng.randint(1000, 9999)}"},
        )
        logs.append(log)
        classifications[i] = classification

    ground_truth = GroundTruth(
        log_classifications=classifications,
        incidents=[],
    )
    return logs, ground_truth


def generate_task2_logs(seed: int = 123) -> Tuple[List[LogEntry], GroundTruth]:
    """Generate 20 logs from 3 sources with 2 incidents for detection."""
    rng = random.Random(seed)

    logs: List[LogEntry] = []
    classifications: Dict[int, str] = {}

    # Incident 1: Database connection exhaustion cascade
    # DB overloaded -> app timeouts -> nginx 502s
    incident1_indices = []

    # Normal background logs
    normal_logs = [
        ("app-server-1", "INFO", "Request processed: GET /api/health 200 5ms", "normal"),
        ("nginx", "INFO", "GET /static/app.js 200 0.002s", "normal"),
        ("postgres-primary", "INFO", "Checkpoint completed: wrote 156 buffers", "normal"),
        ("app-server-1", "INFO", "Cache hit for key: user_session_abc", "normal"),
        ("nginx", "INFO", "GET /api/metrics 200 0.015s", "normal"),
        ("postgres-primary", "INFO", "Autovacuum: processing table public.sessions", "normal"),
    ]

    # Incident 1 logs: DB connection exhaustion
    incident1_logs = [
        ("postgres-primary", "WARN", "Connection count: 490/500 - approaching max_connections limit", "warning"),
        ("postgres-primary", "ERROR", "Too many connections: max_connections(500) reached", "error"),
        ("app-server-1", "ERROR", "Database connection timeout after 30000ms - pool exhausted (0/50 available)", "error"),
        ("app-server-1", "ERROR", "Failed to process request: GET /api/orders - DBConnectionError", "error"),
        ("nginx", "ERROR", "upstream timed out (110: Connection timed out) while connecting to app-server-1", "error"),
        ("nginx", "ERROR", "502 Bad Gateway - upstream returned error for GET /api/orders", "error"),
    ]

    # Incident 2 logs: Disk space issue
    incident2_indices = []
    incident2_logs = [
        ("app-server-1", "WARN", "Log file rotation failed: /var/log/app/access.log - disk full", "warning"),
        ("app-server-1", "ERROR", "Failed to write to disk: No space left on device (errno=28)", "error"),
        ("postgres-primary", "WARN", "WAL segment 000000010000000100000042 could not be archived: disk full", "warning"),
        ("app-server-1", "FATAL", "Application shutdown: unable to write to transaction log", "critical"),
    ]

    # More normal logs
    more_normal = [
        ("nginx", "INFO", "GET /favicon.ico 200 0.001s", "normal"),
        ("app-server-1", "DEBUG", "GC pause: 23ms, heap: 1.2GB/2GB", "normal"),
        ("postgres-primary", "INFO", "Statement duration: 0.045s SELECT count(*) FROM users", "normal"),
        ("nginx", "INFO", "POST /api/login 200 0.125s", "normal"),
    ]

    # Build the log list in a realistic interleaved order
    all_log_groups = [
        (normal_logs[:3], None),
        (incident1_logs[:2], 1),
        (normal_logs[3:5], None),
        (incident1_logs[2:4], 1),
        (incident2_logs[:2], 2),
        (incident1_logs[4:6], 1),
        (normal_logs[5:6], None),
        (incident2_logs[2:4], 2),
        (more_normal, None),
    ]

    idx = 0
    for group, incident_id in all_log_groups:
        for source, level, message, classification in group:
            log = LogEntry(
                timestamp=f"2024-01-15T10:{idx:02d}:{rng.randint(0, 59):02d}Z",
                source=source,
                level=level,
                message=message,
                metadata={"request_id": f"req-{rng.randint(1000, 9999)}"},
            )
            logs.append(log)
            classifications[idx] = classification
            if incident_id == 1:
                incident1_indices.append(idx)
            elif incident_id == 2:
                incident2_indices.append(idx)
            idx += 1

    ground_truth = GroundTruth(
        log_classifications=classifications,
        incidents=[
            {
                "type": "resource_exhaustion",
                "severity": "P2",
                "correlated_indices": incident1_indices,
                "description": "Database connection pool exhaustion causing cascading failures",
            },
            {
                "type": "resource_exhaustion",
                "severity": "P2",
                "correlated_indices": incident2_indices,
                "description": "Disk space exhaustion affecting logging and database WAL",
            },
        ],
    )
    return logs, ground_truth


def generate_task3_logs(seed: int = 456) -> Tuple[List[LogEntry], GroundTruth]:
    """Generate 30+ logs from 5 sources with 3 incidents including a subtle security issue."""
    rng = random.Random(seed)

    logs: List[LogEntry] = []
    classifications: Dict[int, str] = {}
    incident1_indices: List[int] = []  # cascading outage
    incident2_indices: List[int] = []  # memory leak / degradation
    incident3_indices: List[int] = []  # subtle security breach

    sources = ["nginx", "app-server-1", "app-server-2", "postgres-primary", "k8s-scheduler"]

    # All log entries with their incident assignment (None = noise/normal)
    log_sequence: List[Tuple[str, str, str, str, Any]] = [
        # Normal noise
        ("nginx", "INFO", "GET /static/bundle.js 200 0.003s", "normal", None),
        ("app-server-1", "DEBUG", "Cache miss for key: product_catalog_v2", "normal", None),
        ("k8s-scheduler", "INFO", "Pod app-server-2-abc health check passed", "normal", None),
        ("postgres-primary", "INFO", "Checkpoint completed: wrote 89 buffers", "normal", None),

        # Incident 3: Subtle security breach - SQL injection + data exfil
        ("nginx", "INFO", "POST /api/search 200 0.342s from 198.51.100.23", "normal", None),  # looks normal but slow
        ("app-server-1", "WARN", "Unusual query pattern detected: nested UNION SELECT in search parameter", "security", 3),
        ("postgres-primary", "INFO", "Statement duration: 2.145s - complex query on users table", "normal", None),  # red herring - looks normal
        ("app-server-1", "ERROR", "SQL syntax warning: potentially unsafe characters in input field 'q'", "security", 3),

        # Normal noise
        ("k8s-scheduler", "INFO", "Node resource usage: CPU 45%, Memory 62%", "normal", None),
        ("nginx", "INFO", "GET /api/products 200 0.089s", "normal", None),
        ("app-server-2", "DEBUG", "Session validated for user_id=1042", "normal", None),

        # Incident 2: Memory leak causing degradation
        ("app-server-1", "WARN", "Heap usage at 82% (3.28GB/4GB) - approaching limit", "warning", 2),
        ("app-server-1", "WARN", "GC pause duration increasing: 450ms (threshold: 200ms)", "warning", 2),
        ("k8s-scheduler", "INFO", "Pod app-server-1-xyz memory: 3.4GB/4GB - nearing limit", "normal", None),  # noise that correlates

        # More noise
        ("nginx", "INFO", "GET /api/health 200 0.002s", "normal", None),
        ("postgres-primary", "INFO", "Autovacuum: processing table public.order_items", "normal", None),
        ("app-server-2", "INFO", "Request processed: POST /api/cart 200 67ms", "normal", None),

        # Incident 1: Cascading outage - DB replication lag -> app errors -> nginx timeouts
        ("postgres-primary", "WARN", "Replication lag: 15.2s on standby pg-replica-1 (threshold: 5s)", "warning", 1),
        ("postgres-primary", "ERROR", "Replication slot 'replica_1' is lagging behind by 256MB of WAL", "error", 1),
        ("app-server-2", "ERROR", "Read query routed to replica failed: replication lag too high, falling back to primary", "error", 1),
        ("app-server-1", "ERROR", "Connection pool to primary overloaded: 48/50 connections in use after replica failover", "error", 1),

        # Incident 3 continues: data exfiltration
        ("app-server-1", "WARN", "Large response payload: 4.2MB for endpoint /api/search - unusual for this endpoint", "security", 3),
        ("nginx", "WARN", "Response size anomaly: /api/search returned 4.2MB (avg: 12KB) to 198.51.100.23", "security", 3),

        # Incident 2 continues: memory gets worse
        ("app-server-1", "ERROR", "OutOfMemoryError: unable to allocate 256MB for request processing", "critical", 2),
        ("k8s-scheduler", "WARN", "Pod app-server-1-xyz OOMKilled - restarting (restart count: 3)", "warning", 2),

        # Incident 1 continues: cascade
        ("nginx", "ERROR", "upstream timed out (110) while connecting to app-server-1:8080", "error", 1),
        ("nginx", "ERROR", "502 Bad Gateway for 23 requests in last 60 seconds", "error", 1),
        ("k8s-scheduler", "WARN", "Pod app-server-1-xyz not ready - removing from service endpoints", "warning", 1),

        # More noise
        ("app-server-2", "INFO", "Scheduled job completed: cleanup_expired_sessions removed 1,204 rows", "normal", None),
        ("postgres-primary", "INFO", "Statement duration: 0.023s UPDATE sessions SET last_active=NOW()", "normal", None),
        ("nginx", "INFO", "GET /robots.txt 200 0.001s", "normal", None),

        # Incident 3: final evidence
        ("app-server-1", "ERROR", "Rate limit exceeded for IP 198.51.100.23: 847 requests in 60s to /api/search", "security", 3),

        # Final noise
        ("k8s-scheduler", "INFO", "Cluster autoscaler: no scaling needed, all nodes within thresholds", "normal", None),
        ("app-server-2", "DEBUG", "Feature flag 'new_checkout_flow' evaluated: enabled for 15% of users", "normal", None),
    ]

    for i, (source, level, message, classification, incident_id) in enumerate(log_sequence):
        log = LogEntry(
            timestamp=f"2024-01-15T10:{i:02d}:{rng.randint(0, 59):02d}Z",
            source=source,
            level=level,
            message=message,
            metadata={
                "request_id": f"req-{rng.randint(10000, 99999)}",
                "pod_name": f"{source}-{rng.choice(['xyz', 'abc', 'def'])}",
            },
        )
        logs.append(log)
        classifications[i] = classification
        if incident_id == 1:
            incident1_indices.append(i)
        elif incident_id == 2:
            incident2_indices.append(i)
        elif incident_id == 3:
            incident3_indices.append(i)

    ground_truth = GroundTruth(
        log_classifications=classifications,
        incidents=[
            {
                "type": "outage",
                "severity": "P1",
                "correlated_indices": incident1_indices,
                "description": "Cascading outage: DB replication lag -> primary overload -> nginx 502s",
            },
            {
                "type": "degradation",
                "severity": "P2",
                "correlated_indices": incident2_indices,
                "description": "Memory leak causing OOMKills and service degradation",
            },
            {
                "type": "security_breach",
                "severity": "P1",
                "correlated_indices": incident3_indices,
                "description": "SQL injection attack with data exfiltration via /api/search",
            },
        ],
    )
    return logs, ground_truth


GENERATORS = {
    "log_classification": generate_task1_logs,
    "incident_detection": generate_task2_logs,
    "full_triage": generate_task3_logs,
}
