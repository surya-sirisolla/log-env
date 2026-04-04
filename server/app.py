"""Server entry point for multi-mode deployment."""

import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict
from fastapi import Body, FastAPI
from environment import LogSentinelEnv
from tasks import TASKS

app = FastAPI(title="LogSentinel Environment Server")
env = LogSentinelEnv()


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint — environment info."""
    return {
        "name": "LogSentinel",
        "description": "Log analysis and incident classification environment for training AI agents on SRE/DevOps tasks",
        "version": "1.0.0",
        "endpoints": {
            "POST /reset": "Reset environment, returns initial observation",
            "POST /step": "Execute action, returns {observation, reward, done}",
            "GET /state": "Current environment state",
            "GET /health": "Health check",
            "GET /tasks": "List available tasks",
        },
        "status": "healthy",
    }


@app.post("/reset")
async def reset(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Reset the environment and return initial observation."""
    task_name = request.get("task_name")
    return env.reset(task_name=task_name)


@app.post("/step")
async def step(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Take a step with the given action."""
    action_data = request.get("action", {})
    return env.step(action_data)


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """Return current environment state."""
    return env.state


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List available tasks."""
    return {
        "tasks": [
            {
                "name": t.name,
                "description": t.description,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "num_logs": t.num_logs,
            }
            for t in TASKS.values()
        ]
    }


def main():
    """Entry point for [project.scripts] server command."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
