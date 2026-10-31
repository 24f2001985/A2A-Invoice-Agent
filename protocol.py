
import uuid
from datetime import datetime, timezone

TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
}


def new_id():
    return str(uuid.uuid4())


def current_time():
    return datetime.now(timezone.utc).isoformat()


def create_task(task_id=None, context_id=None):
    return {
        "id": task_id or new_id(),
        "contextId": context_id or new_id(),
        "status": {
            "state": "TASK_STATE_SUBMITTED",
            "timestamp": current_time()
        },
        "history": [],
        "artifacts": []
    }


def add_message(task, message):
    task.setdefault("history", []).append(message)
    return task


def set_task_state(task, state):
    current_state = task.get("status", {}).get("state")

    if current_state in TERMINAL_STATES:
        raise ValueError("Terminal task states cannot be changed")

    task["status"] = {
        "state": state,
        "timestamp": current_time()
    }

    return task
