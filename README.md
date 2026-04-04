# LogSentinel — Log Analysis & Incident Classification Environment

An [OpenEnv](https://github.com/facebookresearch/openenv)-compliant environment that simulates real-world **log analysis and incident classification** workflows. AI agents receive raw application/infrastructure logs and must parse them, classify severity, detect incidents by correlating related logs, and recommend remediation actions.

## Motivation

Real SREs and DevOps engineers spend hours triaging logs during incidents — scanning across multiple sources, correlating events, determining severity, and deciding on remediation. LogSentinel provides a structured environment where AI agents can practice and be evaluated on these exact skills, from basic log classification to full multi-source incident triage.

## Tasks

### Task 1: `log_classification` (Easy)
- **10 log entries** from a single source (`app-server-1`)
- Classify each log as: `normal`, `warning`, `error`, `critical`, or `security`
- Straightforward logs with clear keywords
- **Max steps:** 15 | **Expected baseline score:** 0.6–0.8

### Task 2: `incident_detection` (Medium)
- **20 log entries** from 3 sources (`nginx`, `app-server-1`, `postgres-primary`)
- Classify logs AND detect **2 incidents** by correlating across sources
- Example: DB connection exhaustion → app timeouts → nginx 502s
- Assign correct severity (P1–P4)
- **Max steps:** 25 | **Expected baseline score:** 0.4–0.6

### Task 3: `full_triage` (Hard)
- **34 log entries** from 5 sources with noise (debug logs, health checks)
- Detect **3 incidents** including a subtle SQL injection / data exfiltration
- Correlate logs, assign severity, recommend remediation, submit full report
- Includes red herrings and ambiguous logs
- **Max steps:** 35 | **Expected baseline score:** 0.2–0.4

## Action Space

Each action is a JSON object with an `action_type` and associated fields:

| Action Type | Fields | Description |
|---|---|---|
| `classify_log` | `target_log_indices`, `classification` | Classify logs as normal/warning/error/critical/security |
| `detect_incident` | `incident_type`, `correlated_indices` | Detect an incident: outage/degradation/security_breach/resource_exhaustion/config_error |
| `assign_severity` | `severity`, `target_log_indices` | Assign P1–P4 severity |
| `correlate_logs` | `correlated_indices` | Group logs belonging to the same incident |
| `recommend_action` | `recommendation` | Free-text remediation suggestion |
| `submit_report` | `report` | Final structured incident report (ends episode) |

### Example Action
```json
{
  "action_type": "classify_log",
  "target_log_indices": [0, 4, 8],
  "classification": "normal"
}
```

## Observation Space

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `log_entries` | `List[LogEntry]` | Batch of logs with timestamp, source, level, message, metadata |
| `task_description` | `str` | What the agent needs to accomplish |
| `time_window` | `str` | Time range of the logs |
| `remaining_steps` | `int` | Steps left before episode ends |
| `previous_action_result` | `str?` | Feedback from the last action |
| `incident_context` | `dict?` | Accumulated context from previous actions |

### Example Log Entry
```json
{
  "timestamp": "2024-01-15T10:01:23Z",
  "source": "app-server-1",
  "level": "ERROR",
  "message": "Database connection timeout after 30000ms - pool exhausted",
  "metadata": {"request_id": "req-4521"}
}
```

## Reward Function

Rewards are continuous in **[0.0, 1.0]** per step:

| Action | Reward |
|---|---|
| Correct log classification | +0.1 to +0.3 (scaled by difficulty: security > critical > error > warning > normal) |
| Correct incident detection | +0.2 |
| Correct severity assignment | +0.15 (partial credit: +0.05 if one level off) |
| Correct log correlation | Up to +0.2 (F1 score against ground truth) |
| Good remediation recommendation | Up to +0.15 (keyword matching) |
| Final report quality | Up to +0.3 (structure, incident count, summary) |
| Wrong classification | 0.0 |
| Wrong severity | 0.0 (or +0.05 partial credit) |

## Setup

### Docker (recommended)
```bash
docker build -t logsentinel .
docker run -p 7860:7860 logsentinel
```

### Local
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Environment Variables
| Variable | Description | Default |
|---|---|---|
| `MY_ENV_V4_TASK` | Default task on reset | `log_classification` |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model for inference | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace API token | — |
| `IMAGE_NAME` | Docker image for inference | — |

### API Endpoints
```
POST /reset          — Reset environment, returns initial observation
POST /step           — Execute action, returns {observation, reward, done}
GET  /state          — Current environment state
GET  /health         — Health check (200 OK)
GET  /tasks          — List available tasks
```

### Running Tests
```bash
pytest tests/ -v
```

### Running Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export IMAGE_NAME="logsentinel"
python inference.py
```

## Baseline Scores

| Task | Expected Score | Steps |
|---|---|---|
| `log_classification` | 0.6–0.8 | 5–10 |
| `incident_detection` | 0.4–0.6 | 10–15 |
| `full_triage` | 0.2–0.4 | 15–25 |

## Example Episode (log_classification)

```
Step 1: Agent classifies logs [0,4,8] as "normal" → reward=0.30
Step 2: Agent classifies logs [2,6] as "warning" → reward=0.30
Step 3: Agent classifies logs [1,7] as "error" → reward=0.40
Step 4: Agent classifies logs [3,9] as "critical" → reward=0.50
Step 5: Agent classifies log [5] as "security" → reward=0.30
Step 6: Agent submits report → reward=0.25
Total: ~2.05 cumulative reward across 6 steps
```

## Architecture

```
server.py          → FastAPI HTTP server (OpenEnv API)
environment.py     → Core env logic (reset/step/state)
models.py          → Pydantic models (Observation, Action, LogEntry, GroundTruth)
tasks.py           → Task definitions (easy/medium/hard)
log_generator.py   → Deterministic synthetic log generation with ground truth
graders.py         → Scoring logic per action type
inference.py       → Baseline agent using OpenAI-compatible LLM
```
