
import asyncio
import os
import re

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from database import (
    initialize_db,
    get_task,
    list_tasks,
    save_task,
    cancel_task,
    reserve_idempotency,
)

from protocol import create_task, add_message, set_task_state
from ai_engine import analyze_packages
from decision_engine import create_proposal, content_hash


load_dotenv()

app = FastAPI(title="A2A Invoice Action Agent")

API_TOKEN = os.getenv("A2A_API_TOKEN", "")
TEST_TOKEN = os.getenv("A2A_TEST_TOKEN")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

BATCH_MEDIA_TYPE = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-results+json"
PROPOSALS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-receipts+json"
A2A_MEDIA_TYPE = "application/a2a+json"


# --------------------------------------------------
# STARTUP
# --------------------------------------------------

@app.on_event("startup")
def startup():
    initialize_db()


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def authenticate(authorization):

    if not API_TOKEN:
        raise HTTPException(
            500,
            "Server authentication is not configured"
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")

    token = authorization.removeprefix("Bearer ")

    if token == API_TOKEN:
        return "user_main"

    if TEST_TOKEN and token == TEST_TOKEN:
        return "user_test"

    raise HTTPException(401, "Unauthorized")


# --------------------------------------------------
# A2A HEADER VALIDATION
# --------------------------------------------------

def check_version(request):

    if request.headers.get("A2A-Version") != "1.0":
        raise HTTPException(
            400,
            "A2A-Version 1.0 required"
        )


def check_a2a_headers(request):

    check_version(request)

    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";")[0].strip().lower()

    if media_type != A2A_MEDIA_TYPE:
        raise HTTPException(
            400,
            "Content-Type must be application/a2a+json"
        )


# --------------------------------------------------
# REQUEST BODY
# --------------------------------------------------

async def read_a2a_body(request):

    try:
        body = await request.json()

    except Exception:
        raise HTTPException(400, "Invalid JSON request")

    if not isinstance(body, dict):
        raise HTTPException(
            400,
            "Request must be a JSON object"
        )

    message = body.get("message")

    if not isinstance(message, dict):
        raise HTTPException(
            400,
            "Missing or invalid message"
        )

    return body, message


# --------------------------------------------------
# BATCH EXTRACTION
# --------------------------------------------------

def extract_batch(message):

    if message.get("role") != "ROLE_USER":
        raise HTTPException(
            400,
            "Message role must be ROLE_USER"
        )

    parts = message.get("parts")

    if not isinstance(parts, list):
        raise HTTPException(
            400,
            "Message parts must be a list"
        )

    batch_parts = [
        part for part in parts
        if isinstance(part, dict)
        and part.get("mediaType") == BATCH_MEDIA_TYPE
    ]

    if len(batch_parts) != 1:
        raise HTTPException(
            400,
            "Exactly one invoice batch part is required"
        )

    batch = batch_parts[0].get("data")

    if not isinstance(batch, dict):
        raise HTTPException(
            400,
            "Invalid invoice batch data"
        )

    if (
        not isinstance(batch.get("packages"), list)
        or not batch["packages"]
    ):
        raise HTTPException(
            400,
            "Batch must contain invoice packages"
        )

    return batch


# --------------------------------------------------
# INITIAL REQUEST VALIDATION
# --------------------------------------------------

def validate_initial_request(body, message):

    message_id = message.get("messageId")

    if not isinstance(message_id, str) or not message_id.strip():
        raise HTTPException(
            400,
            "messageId is required"
        )

    if message.get("taskId") or message.get("contextId"):
        raise HTTPException(
            400,
            "Initial request must not contain taskId or contextId"
        )

    configuration = body.get("configuration", {})

    if not isinstance(configuration, dict):
        raise HTTPException(
            400,
            "Invalid configuration"
        )

    accepted_modes = configuration.get(
        "acceptedOutputModes",
        []
    )

    if not isinstance(accepted_modes, list):
        raise HTTPException(
            400,
            "Invalid acceptedOutputModes"
        )

    required_modes = {
        PROPOSALS_MEDIA_TYPE,
        RECEIPTS_MEDIA_TYPE
    }

    if not required_modes.issubset(set(accepted_modes)):
        raise HTTPException(
            400,
            "Both output media types must be accepted"
        )

    return message_id


# --------------------------------------------------
# BATCH VALIDATION
# --------------------------------------------------

def validate_batch(batch):

    for field in ("batchId", "policyRevision"):

        if (
            not isinstance(batch.get(field), str)
            or not batch[field].strip()
        ):
            raise HTTPException(
                400,
                f"{field} is required"
            )

    packages = batch["packages"]
    package_ids = []

    for package in packages:

        if not isinstance(package, dict):
            raise HTTPException(
                400,
                "Each package must be an object"
            )

        package_id = package.get("packageId")

        if (
            not isinstance(package_id, str)
            or not package_id.strip()
        ):
            raise HTTPException(
                400,
                "Every package needs a packageId"
            )

        package_ids.append(package_id)

    if len(package_ids) != len(set(package_ids)):
        raise HTTPException(
            400,
            "Duplicate packageId values are not allowed"
        )

    return packages


# --------------------------------------------------
# WAIT FOR AN EXISTING TASK
# --------------------------------------------------

async def wait_for_reserved_task(task_id, principal):

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 45

    while loop.time() < deadline:

        task = get_task(task_id, principal)

        if task is None:
            raise HTTPException(
                500,
                "Reserved task not found"
            )

        if task["status"]["state"] != "TASK_STATE_SUBMITTED":
            return task

        await asyncio.sleep(0.1)

    raise HTTPException(
        503,
        "Task is still processing. Retry shortly."
    )


# --------------------------------------------------
# AGENT CARD
# --------------------------------------------------

@app.get("/.well-known/agent-card.json")
def agent_card():

    return {
        "name": "Invoice Action Agent",

        "description": (
            "Analyzes invoice packages, proposes evidence-based "
            "actions, and executes only approved actions."
        ),

        "version": "1.0.0",

        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True
        },

        "defaultInputModes": [
            BATCH_MEDIA_TYPE
        ],

        "defaultOutputModes": [
            PROPOSALS_MEDIA_TYPE,
            RECEIPTS_MEDIA_TYPE
        ],

        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",

                "description": (
                    "Analyzes invoice evidence and safely "
                    "processes approved invoice actions."
                ),

                "tags": [
                    "invoice",
                    "payments",
                    "approval",
                    "duplicate detection"
                ]
            }
        ],

        "supportedInterfaces": [
            {
                "url": PUBLIC_BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        ]
    }


# --------------------------------------------------
# CONTINUATION / APPROVAL PROCESSING
# --------------------------------------------------

def handle_continuation(message, principal):

    task_id = message.get("taskId")
    context_id = message.get("contextId")

    if not isinstance(task_id, str) or not task_id:
        raise HTTPException(400, "Missing taskId")

    if not isinstance(context_id, str) or not context_id:
        raise HTTPException(400, "Missing contextId")

    task = get_task(task_id, principal)

    if task is None:
        raise HTTPException(404, "Task not found")

    if task.get("contextId") != context_id:
        raise HTTPException(400, "Context ID mismatch")

    if task.get("status", {}).get("state") != "TASK_STATE_INPUT_REQUIRED":
        raise HTTPException(
            409,
            "Task is not awaiting input"
        )

    parts = message.get("parts", [])

    result_parts = [
        part for part in parts
        if isinstance(part, dict)
        and part.get("mediaType") == RESULTS_MEDIA_TYPE
    ]

    if len(result_parts) != 1:
        raise HTTPException(
            400,
            "Exactly one results part is required"
        )

    result_data = result_parts[0].get("data")

    if not isinstance(result_data, dict):
        raise HTTPException(
            400,
            "Invalid results data"
        )

    proposal_artifact = next(
        (
            artifact
            for artifact in task.get("artifacts", [])
            if any(
                part.get("mediaType") == PROPOSALS_MEDIA_TYPE
                for part in artifact.get("parts", [])
            )
        ),
        None
    )

    if proposal_artifact is None:
        raise HTTPException(
            500,
            "Proposal artifact missing"
        )

    proposal_part = next(
        part
        for part in proposal_artifact["parts"]
        if part.get("mediaType") == PROPOSALS_MEDIA_TYPE
    )

    proposal_data = proposal_part["data"]

    if result_data.get("batchId") != proposal_data.get("batchId"):
        raise HTTPException(
            400,
            "Batch ID mismatch"
        )

    proposals = {
        item["packageId"]: item
        for item in proposal_data["proposals"]
    }

    results = result_data.get("results")

    if (
        not isinstance(results, list)
        or len(results) != len(proposals)
    ):
        raise HTTPException(
            400,
            "Results must match all proposals"
        )

    seen = set()
    executions = []

    for result in results:

        if not isinstance(result, dict):
            raise HTTPException(
                400,
                "Invalid package result"
            )

        package_id = result.get("packageId")

        if package_id not in proposals or package_id in seen:
            raise HTTPException(
                400,
                "Invalid or duplicate package result"
            )

        seen.add(package_id)

        proposal = proposals[package_id]

        if result.get("actionId") != proposal["actionId"]:
            raise HTTPException(
                400,
                "Action ID mismatch"
            )

        if result.get("action") != proposal["action"]:
            raise HTTPException(
                400,
                "Action mismatch"
            )

        if result.get("outcome") not in ("ACCEPTED", "REJECTED"):
            raise HTTPException(
                400,
                "Invalid outcome"
            )

        nonce = result.get("receiptNonce")

        if not isinstance(nonce, str) or not nonce.strip():
            raise HTTPException(
                400,
                "Missing receipt nonce"
            )

        # Execute only accepted actions.
        if result["outcome"] == "ACCEPTED":

            executions.append({
                "packageId": package_id,
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "receiptNonce": nonce,
                "facts": proposal["facts"],
                "evidenceRefs": proposal["evidenceRefs"]
            })

    add_message(task, message)

    # Preserve the original proposal artifact.
    # Add one receipt artifact.
    task["artifacts"].append({
        "artifactId": create_task()["id"],

        "parts": [
            {
                "mediaType": RECEIPTS_MEDIA_TYPE,

                "data": {
                    "batchId": proposal_data["batchId"],
                    "executions": executions
                }
            }
        ]
    })

    set_task_state(
        task,
        "TASK_STATE_COMPLETED"
    )

    save_task(
        task["id"],
        principal,
        task["status"]["state"],
        task
    )

    return task


# --------------------------------------------------
# MAIN A2A MESSAGE ENDPOINT
# --------------------------------------------------

@app.post("/message:send")
async def send_message(request: Request):

    check_a2a_headers(request)

    # Authenticate and get the internal user identity.
    principal = authenticate(
        request.headers.get("Authorization")
    )

    body, message = await read_a2a_body(request)

    # Handle continuation messages.
    if message.get("taskId"):

        if message.get("role") != "ROLE_USER":
            raise HTTPException(
                400,
                "Invalid message role"
            )

        task = handle_continuation(
            message,
            principal
        )

        return JSONResponse(
            content={"task": task},
            media_type=A2A_MEDIA_TYPE
        )

    # Validate initial invoice submission.
    message_id = validate_initial_request(
        body,
        message
    )

    batch = extract_batch(message)
    validate_batch(batch)

    message_hash = content_hash(message)

    task_id = create_task()["id"]

    # Reserve the message ID and task atomically.
    try:

        reservation = reserve_idempotency(
            principal,
            message_id,
            message_hash,
            task_id
        )

    except ValueError:

        raise HTTPException(
            409,
            detail={
                "code": "IDEMPOTENCY_CONFLICT"
            }
        )

    # If this message was already submitted,
    # return its existing task.
    if not reservation["created"]:

        existing_task = await wait_for_reserved_task(
            reservation["task_id"],
            principal
        )

        return JSONResponse(
            content={"task": existing_task},
            media_type=A2A_MEDIA_TYPE
        )

    placeholder = get_task(
        task_id,
        principal
    )

    if placeholder is None:
        raise HTTPException(
            500,
            "Reserved task not found"
        )

    # IMPORTANT:
    # AI processing must be outside the previous
    # if block. This was the indentation bug.
    try:

        artifact = await build_proposal_artifact(batch)

        task = create_task(
            task_id=task_id,
            context_id=placeholder["contextId"]
        )

        add_message(task, message)

        task["artifacts"].append(artifact)

        set_task_state(
            task,
            "TASK_STATE_INPUT_REQUIRED"
        )

        save_task(
            task["id"],
            principal,
            task["status"]["state"],
            task
        )

        return JSONResponse(
            content={"task": task},
            media_type=A2A_MEDIA_TYPE
        )

    except HTTPException:
        raise

    except ValueError as error:

        print("Invoice validation error:", str(error))

        raise HTTPException(
            422,
            str(error)
        )

    except Exception:

        import traceback
        traceback.print_exc()

        raise HTTPException(
            500,
            "Invoice processing failed"
        )


# --------------------------------------------------
# LIST TASKS
# --------------------------------------------------

@app.get("/tasks")
def get_tasks(
    request: Request,
    authorization: str = Header(None)
):

    check_version(request)

    principal = authenticate(authorization)

    return {
        "tasks": list_tasks(principal)
    }


# --------------------------------------------------
# GET INDIVIDUAL TASK
# --------------------------------------------------

@app.get("/tasks/{task_id}")
def get_task_route(
    task_id: str,
    request: Request,
    authorization: str = Header(None)
):

    check_version(request)

    principal = authenticate(authorization)

    task = get_task(
        task_id,
        principal
    )

    if task is None:
        raise HTTPException(
            404,
            "Task not found"
        )

    return task


# --------------------------------------------------
# CANCEL TASK
# --------------------------------------------------

@app.post("/tasks/{task_id}:cancel")
def cancel_task_route(
    task_id: str,
    request: Request,
    authorization: str = Header(None)
):

    check_version(request)

    principal = authenticate(authorization)

    try:

        task = cancel_task(
            task_id,
            principal
        )

        if task is None:
            raise HTTPException(
                404,
                "Task not found"
            )

        return JSONResponse(
            content={"task": task},
            media_type=A2A_MEDIA_TYPE
        )

    except HTTPException:
        raise

    except ValueError as error:

        raise HTTPException(
            409,
            str(error)
        )


# --------------------------------------------------
# EVIDENCE REFERENCE EXTRACTION
# --------------------------------------------------

def collect_evidence_refs(package):

    refs = set()

    documents = package.get("documents", [])

    for document in documents:

        if isinstance(document, str):

            refs.update(
                re.findall(
                    r"\[[^\[\]]+\]",
                    document
                )
            )

    return sorted(refs)


# --------------------------------------------------
# AI PROPOSAL GENERATION
# --------------------------------------------------

async def build_proposal_artifact(batch):

    ai_result = await analyze_packages(batch)

    proposals_by_id = {
        item["packageId"]: item
        for item in ai_result["proposals"]
    }

    proposals = []

    for package in batch["packages"]:

        package_id = package["packageId"]

        decision = proposals_by_id.get(package_id)

        if decision is None:
            raise ValueError(
                f"Missing decision for {package_id}"
            )

        proposal = create_proposal(
            package_id,
            decision,
            collect_evidence_refs(package)
        )

        proposals.append(proposal)

    return {
        "artifactId": create_task()["id"],

        "parts": [
            {
                "mediaType": PROPOSALS_MEDIA_TYPE,

                "data": {
                    "batchId": batch["batchId"],
                    "proposals": proposals
                }
            }
        ]
    }
