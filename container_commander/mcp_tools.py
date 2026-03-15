"""
Container Commander — MCP Tools for KI Integration
═══════════════════════════════════════════════════════
Registers Container Commander functions as MCP tools
so the KI can request containers, execute code, manage snapshots.

KI System Prompt Injection:
  Every container-related response includes the reminder:
  "Sei ressourcenschonend. Beende Container sofort nach Erledigung."

Tool List:
  - request_container    → Deploy a blueprint as container
  - stop_container       → Stop a running container
  - exec_in_container    → Run a command inside container
  - container_logs       → Get logs from container
  - container_stats      → Get CPU/RAM/efficiency score
  - blueprint_list       → List available blueprints
  - blueprint_get        → Get blueprint details
  - blueprint_create     → Create a new blueprint (trusted images only)
  - snapshot_list        → List volume snapshots
  - snapshot_restore     → Restore a snapshot
  - optimize_container   → Adjust resource limits on running container
  - home_write           → Write file to TRION home container
  - home_read            → Read file from TRION home container
  - home_list            → List directory in TRION home container
  - container_list       → List all active TRION containers (discovery)
  - container_inspect    → Detailed info about a specific container
  - autonomy_cron_*      → Manage autonomous cron schedules (user/TRION)
  - cron_reference_links_list → Read-only GitHub inspiration collections
  - storage_provision_container → Atomic: scope + blueprint + optional start
"""

import logging
import os
import asyncio
import threading
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ── KI System Prompt ──────────────────────────────────────
# NOTE: CONTAINER_SYSTEM_PROMPT is injected via persona.py (CONTAINER-MANAGEMENT section),
# not from here. get_system_prompt() is kept for API compatibility but persona.py is authoritative.

CONTAINER_SYSTEM_PROMPT = """Du hast Zugriff auf Container-Management Tools.
- Starte nur Container die du wirklich brauchst.
- Führe ALLE Schritte einer Aufgabe im selben Container aus — stoppe erst nach Abschluss der gesamten Pipeline.
- Nutze die kleinsten Resource-Limits die ausreichen.
- Bei efficiency_level "red" → optimize oder stoppen, aber nur wenn kein aktiver Task läuft."""

# These tools are provided by Fast-Lane at runtime and should usually be hidden
# from Commander MCP registration to avoid duplicate registry entries.
FAST_LANE_TOOL_NAMES = {"home_read", "home_write", "home_list"}


# ── Tool Definitions (MCP Schema) ─────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "request_container",
        "description": "Deploy a container from a blueprint. Returns container_id plus connection hints (IP/ports). Keep the container running across all steps of a multi-step task — stop it only when the entire task is complete.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "blueprint_id": {"type": "string", "description": "ID of the blueprint to deploy (use blueprint_list to see available)"},
                "resume_volume": {"type": "string", "description": "Optional: volume name to resume previous workspace"},
            },
            "required": ["blueprint_id"]
        }
    },
    {
        "name": "stop_container",
        "description": "Stop and remove a running container. Call this only after the complete task pipeline is finished — not between steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID to stop"}
            },
            "required": ["container_id"]
        }
    },
    {
        "name": "exec_in_container",
        "description": "Execute a shell command inside a running container. Returns {exit_code, stdout, stderr, truncated}. Only commands in the blueprint's allowed_exec list are permitted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID"},
                "command": {"type": "string", "description": "Shell command to execute (e.g. 'python3 script.py')"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)", "default": 30}
            },
            "required": ["container_id", "command"]
        }
    },
    {
        "name": "container_logs",
        "description": "Get recent logs from a container.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID"},
                "tail": {"type": "integer", "description": "Number of lines (default: 100)", "default": 100}
            },
            "required": ["container_id"]
        }
    },
    {
        "name": "container_stats",
        "description": "Get live resource stats (CPU, RAM, network) and efficiency score. Check this regularly! Red efficiency = stop or optimize.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID"}
            },
            "required": ["container_id"]
        }
    },
    {
        "name": "blueprint_list",
        "description": "List all available container blueprints. Use this to find the right blueprint before requesting a container.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Optional tag filter (e.g. 'python', 'data')"}
            }
        }
    },
    {
        "name": "blueprint_get",
        "description": "Get detailed information about a specific blueprint including Dockerfile, resources, and required secrets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "blueprint_id": {"type": "string", "description": "Blueprint ID to look up"}
            },
            "required": ["blueprint_id"]
        }
    },
    {
        "name": "storage_scope_list",
        "description": "List approved storage scopes for host bind mounts.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "storage_scope_upsert",
        "description": "Create or update a storage scope with approved host roots.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Scope name (e.g. 'gaming')"},
                "roots": {
                    "type": "array",
                    "description": "Allowed roots: [{path:'/data/games', mode:'rw'}]",
                    "items": {"type": "object"}
                },
                "approved_by": {"type": "string", "description": "Actor marker", "default": "user"}
            },
            "required": ["name", "roots"]
        }
    },
    {
        "name": "storage_scope_delete",
        "description": "Delete an existing storage scope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Scope name"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "snapshot_list",
        "description": "List available snapshots for workspace recovery.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "volume_name": {"type": "string", "description": "Optional: filter by volume name"}
            }
        }
    },
    {
        "name": "snapshot_restore",
        "description": "Restore a snapshot into a volume. Use with request_container(resume_volume=...) to continue previous work.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Snapshot filename (from snapshot_list)"},
                "target_volume": {"type": "string", "description": "Optional: target volume name (auto-generated if empty)"}
            },
            "required": ["filename"]
        }
    },
    {
        "name": "optimize_container",
        "description": "Adjust resource limits on a running container. Use when efficiency is yellow/red to right-size resources.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID"},
                "cpu_limit": {"type": "string", "description": "New CPU limit (e.g. '0.5')"},
                "memory_limit": {"type": "string", "description": "New memory limit (e.g. '256m')"}
            },
            "required": ["container_id"]
        }
    },
    {
        "name": "blueprint_create",
        "description": "Create a new container blueprint from a trusted Docker image. Only official and verified images are allowed (ubuntu, python, node, postgres, mongo, redis, nginx, alpine).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique blueprint ID (e.g. 'my-python-sandbox')"},
                "image": {"type": "string", "description": "Docker image (e.g. 'python:3.12-slim')"},
                "name": {"type": "string", "description": "Human-readable name"},
                "description": {"type": "string", "description": "What this blueprint is for"},
                "memory_limit": {"type": "string", "description": "Memory limit (e.g. '512m')", "default": "512m"},
                "cpu_limit": {"type": "string", "description": "CPU limit (e.g. '0.5')", "default": "0.5"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for filtering"},
                "network": {
                    "type": "string",
                    "description": "Network mode",
                    "enum": ["none", "internal", "bridge", "full"],
                    "default": "internal",
                },
                "ports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Port mappings (e.g. ['47984:47984/tcp']).",
                },
                "runtime": {
                    "type": "string",
                    "description": "Container runtime (e.g. 'nvidia').",
                },
                "devices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Device mappings (e.g. ['/dev/dri:/dev/dri']).",
                },
                "environment": {
                    "type": "object",
                    "description": "Static env vars. Supports secret refs via vault://SECRET_NAME values.",
                    "additionalProperties": {"type": "string"},
                },
                "storage_scope": {
                    "type": "string",
                    "description": "Optional approved storage scope name for bind mounts.",
                },
                "mounts": {
                    "type": "array",
                    "description": (
                        "Host→container mount mappings. Each entry: "
                        "{host: '/abs/path', container: '/app/data', mode: 'rw', type: 'bind'}. "
                        "Bind-mount host paths must be within an approved storage_scope. "
                        "Use type='volume' for named Docker volumes (host = volume name)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "host":      {"type": "string", "description": "Absolute host path or volume name"},
                            "container": {"type": "string", "description": "Mount target inside container"},
                            "mode":      {"type": "string", "enum": ["ro", "rw"], "default": "rw"},
                            "type":      {"type": "string", "enum": ["bind", "volume"], "default": "bind"},
                        },
                        "required": ["host", "container"],
                    },
                },
                "healthcheck": {
                    "type": "object",
                    "description": "Healthcheck config object.",
                },
                "cap_add": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Linux capabilities to add (e.g. ['NET_ADMIN']).",
                },
                "shm_size": {
                    "type": "string",
                    "description": "Shared memory size (e.g. '1g').",
                },
                "allowed_exec": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Allowed command prefixes for exec_in_container (e.g. ['python', 'sh']). Empty = no restriction."
                }
            },
            "required": ["id", "image", "name"]
        }
    },
    {
        "name": "home_write",
        "description": "Write a file to TRION's home container. Creates the home container if not running.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to /home/trion (e.g. 'notes/todo.md')"},
                "content": {"type": "string", "description": "File content to write"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "home_read",
        "description": "Read a file from TRION's home container.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to /home/trion (e.g. 'notes/todo.md')"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "home_list",
        "description": "List contents of a directory in TRION's home container.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to /home/trion (default: root)", "default": "."}
            }
        }
    },
    {
        "name": "container_list",
        "description": "List all active TRION-managed containers with their status, blueprint, and TTL. Use this to discover running containers before exec or stats.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "Filter by status: 'running', 'stopped', or 'all' (default: 'all')",
                    "enum": ["running", "stopped", "all"]
                }
            }
        }
    },
    {
        "name": "container_inspect",
        "description": "Get detailed information about a specific container: image, network, resource limits, mounts, TTL remaining. Use after container_list to get the container_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID (from container_list or request_container)"}
            },
            "required": ["container_id"]
        }
    },
    {
        "name": "autonomy_cron_status",
        "description": "Get scheduler status and high-level counts for autonomy cron jobs.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "autonomy_cron_list_jobs",
        "description": "List all autonomy cron jobs with runtime state and next execution time.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "autonomy_cron_validate",
        "description": "Validate a 5-field cron expression.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cron": {"type": "string", "description": "Cron expression (min hour day month weekday)"}
            },
            "required": ["cron"]
        }
    },
    {
        "name": "autonomy_cron_create_job",
        "description": "Create a new autonomous cron job. Supports schedule_mode=recurring (cron) or schedule_mode=one_shot (run_at). TRION-created jobs are guarded by safety policy and may require explicit user approval for risky objectives.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Human-readable job name"},
                "objective": {"type": "string", "description": "Autonomous objective to run"},
                "conversation_id": {"type": "string", "description": "Conversation scope"},
                "cron": {"type": "string", "description": "Cron expression (min hour day month weekday)"},
                "schedule_mode": {"type": "string", "enum": ["recurring", "one_shot"], "default": "recurring"},
                "run_at": {"type": "string", "description": "UTC timestamp for one-shot execution (ISO 8601)"},
                "timezone": {"type": "string", "description": "Timezone (e.g. UTC, Europe/Berlin)", "default": "UTC"},
                "max_loops": {"type": "integer", "description": "Autonomy max loops", "default": 10},
                "created_by": {"type": "string", "description": "actor marker (user|trion)", "default": "user"},
                "user_approved": {"type": "boolean", "description": "Set true only when a user explicitly approved a risky TRION objective", "default": False},
                "enabled": {"type": "boolean", "description": "Whether job starts active", "default": True}
            },
            "required": ["name", "objective"]
        }
    },
    {
        "name": "autonomy_cron_update_job",
        "description": "Update fields of an existing autonomy cron job, including schedule_mode and one-shot run_at.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cron_job_id": {"type": "string", "description": "Cron job ID"},
                "name": {"type": "string"},
                "objective": {"type": "string"},
                "conversation_id": {"type": "string"},
                "cron": {"type": "string"},
                "schedule_mode": {"type": "string", "enum": ["recurring", "one_shot"]},
                "run_at": {"type": "string", "description": "UTC timestamp for one-shot execution (ISO 8601)"},
                "timezone": {"type": "string"},
                "max_loops": {"type": "integer"},
                "created_by": {"type": "string"},
                "user_approved": {"type": "boolean"},
                "enabled": {"type": "boolean"}
            },
            "required": ["cron_job_id"]
        }
    },
    {
        "name": "autonomy_cron_pause_job",
        "description": "Pause a cron job (keeps definition but stops scheduling).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cron_job_id": {"type": "string", "description": "Cron job ID"}
            },
            "required": ["cron_job_id"]
        }
    },
    {
        "name": "autonomy_cron_resume_job",
        "description": "Resume a paused cron job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cron_job_id": {"type": "string", "description": "Cron job ID"}
            },
            "required": ["cron_job_id"]
        }
    },
    {
        "name": "autonomy_cron_run_now",
        "description": "Manually enqueue a run for a cron job immediately.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cron_job_id": {"type": "string", "description": "Cron job ID"}
            },
            "required": ["cron_job_id"]
        }
    },
    {
        "name": "autonomy_cron_delete_job",
        "description": "Delete a cron job definition permanently.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cron_job_id": {"type": "string", "description": "Cron job ID"}
            },
            "required": ["cron_job_id"]
        }
    },
    {
        "name": "autonomy_cron_queue",
        "description": "Inspect cron queue state: pending, running, recent dispatches.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "cron_reference_links_list",
        "description": "Read configured GitHub reference collections (read-only) for cronjobs, skills, and blueprints.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter",
                    "enum": ["cronjobs", "skills", "blueprints"]
                },
                "include_disabled": {
                    "type": "boolean",
                    "description": "Include disabled links",
                    "default": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Max links to return per category (1-100)",
                    "default": 50
                }
            }
        }
    },
    {
        "name": "storage_provision_container",
        "description": (
            "Atomic: create the host storage directory (if needed), register it as an approved "
            "storage scope, create a blueprint with the correct host→container mount mapping, "
            "and optionally start the container. "
            "Use this instead of the manual chain "
            "(storage_create_service_dir → storage_scope_upsert → blueprint_create → request_container). "
            "The directory is auto-created with mode 0o750 if it does not exist — "
            "use owner_uid/owner_gid so non-root container users can write to it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "blueprint_id": {
                    "type": "string",
                    "description": "Unique blueprint ID (e.g. 'my-python-service'). Also used as scope name.",
                },
                "image": {
                    "type": "string",
                    "description": "Trusted Docker image (e.g. 'python:3.12-slim').",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable blueprint name.",
                },
                "storage_host_path": {
                    "type": "string",
                    "description": (
                        "Absolute host path to mount (e.g. '/data/services/my-app'). "
                        "Auto-created with mode 0o750 if it does not exist."
                    ),
                },
                "container_mount_path": {
                    "type": "string",
                    "description": "Where to mount the path inside the container (e.g. '/app/data').",
                    "default": "/app/data",
                },
                "storage_mode": {
                    "type": "string",
                    "enum": ["ro", "rw"],
                    "description": "Mount mode: 'rw' (read-write, default) or 'ro' (read-only).",
                    "default": "rw",
                },
                "owner_uid": {
                    "type": "integer",
                    "description": (
                        "UID to chown the directory to after creation. "
                        "Set to the container's service user UID so non-root containers can write. "
                        "Only applied when the directory is newly created. "
                        "Example: 1000 for a typical non-root app user."
                    ),
                },
                "owner_gid": {
                    "type": "integer",
                    "description": (
                        "GID to chown the directory to after creation (defaults to owner_uid if omitted). "
                        "Only applied when the directory is newly created."
                    ),
                },
                "description": {"type": "string", "description": "What this blueprint is for."},
                "memory_limit": {"type": "string", "description": "Memory limit (e.g. '512m').", "default": "512m"},
                "cpu_limit": {"type": "string", "description": "CPU limit (e.g. '0.5').", "default": "0.5"},
                "network": {
                    "type": "string",
                    "enum": ["none", "internal", "bridge", "full"],
                    "default": "internal",
                },
                "environment": {
                    "type": "object",
                    "description": "Static env vars (supports vault://SECRET_NAME refs).",
                    "additionalProperties": {"type": "string"},
                },
                "allowed_exec": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Allowed command prefixes for exec_in_container.",
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "auto_start": {
                    "type": "boolean",
                    "description": "If true, start the container immediately after creation (default: true).",
                    "default": True,
                },
            },
            "required": ["blueprint_id", "image", "name", "storage_host_path"],
        },
    },
]


# ── Tool Execution ────────────────────────────────────────

def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a container commander tool and return the result."""
    
    try:
        if tool_name == "request_container":
            return _tool_request_container(arguments)
        elif tool_name == "stop_container":
            return _tool_stop_container(arguments)
        elif tool_name == "exec_in_container":
            return _tool_exec(arguments)
        elif tool_name == "container_logs":
            return _tool_logs(arguments)
        elif tool_name == "container_stats":
            return _tool_stats(arguments)
        elif tool_name == "blueprint_list":
            return _tool_blueprint_list(arguments)
        elif tool_name == "blueprint_get":
            return _tool_blueprint_get(arguments)
        elif tool_name == "storage_scope_list":
            return _tool_storage_scope_list(arguments)
        elif tool_name == "storage_scope_upsert":
            return _tool_storage_scope_upsert(arguments)
        elif tool_name == "storage_scope_delete":
            return _tool_storage_scope_delete(arguments)
        elif tool_name == "snapshot_list":
            return _tool_snapshot_list(arguments)
        elif tool_name == "snapshot_restore":
            return _tool_snapshot_restore(arguments)
        elif tool_name == "optimize_container":
            return _tool_optimize(arguments)
        elif tool_name == "blueprint_create":
            return _tool_blueprint_create(arguments)
        elif tool_name == "home_write":
            return _tool_home_write(arguments)
        elif tool_name == "home_read":
            return _tool_home_read(arguments)
        elif tool_name == "home_list":
            return _tool_home_list(arguments)
        elif tool_name == "container_list":
            return _tool_container_list(arguments)
        elif tool_name == "container_inspect":
            return _tool_container_inspect(arguments)
        elif tool_name == "autonomy_cron_status":
            return _tool_autonomy_cron_status(arguments)
        elif tool_name == "autonomy_cron_list_jobs":
            return _tool_autonomy_cron_list_jobs(arguments)
        elif tool_name == "autonomy_cron_validate":
            return _tool_autonomy_cron_validate(arguments)
        elif tool_name == "autonomy_cron_create_job":
            return _tool_autonomy_cron_create_job(arguments)
        elif tool_name == "autonomy_cron_update_job":
            return _tool_autonomy_cron_update_job(arguments)
        elif tool_name == "autonomy_cron_pause_job":
            return _tool_autonomy_cron_pause_job(arguments)
        elif tool_name == "autonomy_cron_resume_job":
            return _tool_autonomy_cron_resume_job(arguments)
        elif tool_name == "autonomy_cron_run_now":
            return _tool_autonomy_cron_run_now(arguments)
        elif tool_name == "autonomy_cron_delete_job":
            return _tool_autonomy_cron_delete_job(arguments)
        elif tool_name == "autonomy_cron_queue":
            return _tool_autonomy_cron_queue(arguments)
        elif tool_name == "cron_reference_links_list":
            return _tool_cron_reference_links_list(arguments)
        elif tool_name == "storage_provision_container":
            return _tool_storage_provision_container(arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"[MCP-Commander] Tool '{tool_name}' failed: {e}")
        return {"error": str(e)}


# ── Tool Implementations ──────────────────────────────────

def _classify_deploy_runtime_error(error: str) -> str:
    msg = str(error or "").strip().lower()
    if msg.startswith("healthcheck_timeout_auto_stopped"):
        return "healthcheck_timeout"
    if msg.startswith("healthcheck_unhealthy_auto_stopped"):
        return "healthcheck_unhealthy"
    if msg.startswith("container_exited_before_ready_auto_stopped"):
        return "container_not_ready"
    return "deploy_conflict"


def _tool_request_container(args: dict) -> dict:
    from .engine import start_container, inspect_container, PendingApprovalError
    blueprint_id = str(args.get("blueprint_id", "")).strip()
    override_resources = None
    if blueprint_id in {"gaming-station", "steam-headless", "gaming_station"}:
        _ensure_gaming_station_blueprint()
        override_resources = _compute_gaming_override_resources()
    try:
        instance = start_container(
            blueprint_id=blueprint_id or args["blueprint_id"],
            override_resources=override_resources,
            resume_volume=args.get("resume_volume"),
            session_id=args.get("session_id", ""),
            conversation_id=args.get("conversation_id", ""),
        )
        details = inspect_container(instance.container_id)
        result = {
            "status": "running",
            "container_id": instance.container_id,
            "name": instance.name,
            "blueprint_id": instance.blueprint_id,
            "volume": instance.volume_name,
            "ttl_seconds": instance.ttl_remaining,
            "ip_address": details.get("ip_address", ""),
            "ports": details.get("ports", []),
            "connection": details.get("connection", {}),
            "hint": "Container läuft. Führe alle Schritte der Aufgabe darin aus und stoppe ihn erst am Ende.",
        }
        if override_resources is not None:
            result["resource_profile"] = {
                "cpu_limit": override_resources.cpu_limit,
                "memory_limit": override_resources.memory_limit,
                "memory_swap": override_resources.memory_swap,
            }
        return result
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    except PendingApprovalError as e:
        return {
            "status": "pending_approval",
            "approval_id": e.approval_id,
            "reason": e.reason,
            "hint": "Der User muss die Netzwerk-Freigabe erst genehmigen.",
        }
    except RuntimeError as e:
        msg = str(e)
        return {
            "status": "error",
            "error": msg,
            "error_code": _classify_deploy_runtime_error(msg),
        }


def _compute_gaming_override_resources():
    """
    Derive a quota-compatible resource profile for gaming requests.
    This keeps request_container deterministic even when default gaming
    blueprint resources exceed current commander quotas.
    """
    from .engine import get_quota
    from .models import ResourceLimits

    try:
        quota = get_quota()
        max_mem = int(getattr(quota, "max_total_memory_mb", 0) or 0)
        used_mem = float(getattr(quota, "memory_used_mb", 0) or 0.0)
        max_cpu = float(getattr(quota, "max_total_cpu", 0) or 0.0)
        used_cpu = float(getattr(quota, "cpu_used", 0) or 0.0)
    except Exception:
        return None

    if max_mem <= 0 or max_cpu <= 0:
        return None

    # Keep small safety headroom to reduce oversubscription spikes.
    mem_headroom = max(0, int(max_mem - used_mem) - 256)
    cpu_headroom = max(0.0, max_cpu - used_cpu - 0.25)

    if mem_headroom < 512 or cpu_headroom < 0.5:
        return None

    mem_limit_mb = min(1536, mem_headroom)
    cpu_limit = min(1.5, cpu_headroom)
    swap_mb = max(1024, min(mem_limit_mb * 2, 4096))

    return ResourceLimits(
        cpu_limit=f"{cpu_limit:.2f}".rstrip("0").rstrip("."),
        memory_limit=f"{int(mem_limit_mb)}m",
        memory_swap=f"{int(swap_mb)}m",
        timeout_seconds=0,
        pids_limit=512,
    )


def _ensure_gaming_station_blueprint() -> None:
    """
    Ensure a deterministic Steam/Sunshine gaming blueprint exists.
    Keeps implementation prompt-free by using fixed defaults.
    """
    from .blueprint_store import get_blueprint, create_blueprint, update_blueprint
    from .models import Blueprint, ResourceLimits, MountDef, NetworkMode

    bp_id = "gaming-station"
    image_ref = "josh5/steam-headless:latest"
    existing = get_blueprint(bp_id)
    if existing:
        legacy_image = str(existing.image or "").strip().lower()
        if legacy_image in {
            "ghcr.io/linuxserver/steam-headless:latest",
            "lscr.io/linuxserver/steam-headless:latest",
        }:
            update_blueprint(
                bp_id,
                {
                    "image": image_ref,
                },
            )
        return

    bp = Blueprint(
        id=bp_id,
        name="Gaming Station (Steam Headless + Sunshine)",
        description="GPU gaming container with Sunshine streaming and Steam session support.",
        image=image_ref,
        resources=ResourceLimits(memory_limit="8g", cpu_limit="4.0", timeout_seconds=0, pids_limit=512),
        mounts=[
            MountDef(host="gaming_steam_config", container="/config", type="volume", mode="rw"),
            MountDef(host="gaming_steam_data", container="/data", type="volume", mode="rw"),
        ],
        network=NetworkMode.FULL,
        ports=[
            "47984:47984/tcp",
            "47989:47989/tcp",
            "48010:48010/tcp",
            "48100-48110:48100-48110/udp",
        ],
        runtime="nvidia",
        environment={
            "TZ": "UTC",
            "PUID": "1000",
            "PGID": "1000",
            "STEAM_USER": "vault://STEAM_USERNAME",
            "STEAM_PASS": "vault://STEAM_PASSWORD",
        },
        healthcheck={"test": "curl -fsS http://127.0.0.1:47989/ || exit 1", "interval_seconds": 30, "timeout_seconds": 5, "retries": 5},
        tags=["gaming", "steam", "sunshine", "gpu", "nvidia"],
        icon="🎮",
    )
    create_blueprint(bp)


def _tool_stop_container(args: dict) -> dict:
    from .engine import stop_container, get_client
    container_id = args["container_id"]
    # Read blueprint_id from Docker labels BEFORE stopping (labels lost after remove)
    blueprint_id = "unknown"
    try:
        _c = get_client().containers.get(container_id)
        blueprint_id = _c.labels.get("trion.blueprint", "unknown")
    except Exception:
        pass
    stopped = stop_container(container_id)
    return {
        "stopped": stopped,
        "container_id": container_id,
        "blueprint_id": blueprint_id,
    }


def _tool_exec(args: dict) -> dict:
    """
    Execute a command in a container.
    Returns structured output: {exit_code, stdout, stderr, truncated, container_id}
    On policy violation: {error: policy_denied, reason, allowed_exec, hint}
    """
    from .engine import exec_in_container_detailed, PolicyViolationError
    try:
        result = exec_in_container_detailed(
            args["container_id"],
            args["command"],
            args.get("timeout", 30),
        )
        # Add truncation notice to stderr if output was cut
        if result.get("truncated"):
            result["stderr"] = (result["stderr"] + "\n[OUTPUT TRUNCATED — max 8000 chars per stream]").strip()
        return result
    except PolicyViolationError as e:
        return {
            "error": "policy_denied",
            "reason": str(e),
            "command": args.get("command", ""),
            "allowed_exec": e.allowed,
            "container_id": args.get("container_id", ""),
            "hint": f"Dieser Befehl ist für Blueprint '{e.blueprint_id}' nicht erlaubt. "
                    f"Erlaubt: {', '.join(e.allowed)}",
        }


def _tool_logs(args: dict) -> dict:
    from .engine import get_container_logs
    logs = get_container_logs(args["container_id"], args.get("tail", 100))
    return {
        "logs": logs,
        "container_id": args["container_id"],
    }


def _tool_stats(args: dict) -> dict:
    from .engine import get_container_stats
    stats = get_container_stats(args["container_id"])
    # Add optimization hint
    eff = stats.get("efficiency", {})
    if eff.get("level") == "red":
        stats["hint"] = "⚠️ Efficiency ist ROT — Container idle? Stoppe ihn oder nutze optimize_container."
    elif eff.get("level") == "yellow":
        stats["hint"] = "Container underutilized. Prüfe ob du kleinere Resources nutzen kannst."
    return stats


def _tool_blueprint_list(args: dict) -> dict:
    from .blueprint_store import list_blueprints
    bps = list_blueprints(tag=args.get("tag"))
    return {
        "blueprints": [
            {
                "id": bp.id,
                "name": bp.name,
                "description": bp.description,
                "icon": bp.icon,
                "network": bp.network.value,
                "tags": bp.tags,
                "resources": {
                    "cpu": bp.resources.cpu_limit,
                    "memory": bp.resources.memory_limit,
                    "timeout": bp.resources.timeout_seconds,
                },
            }
            for bp in bps
        ],
        "count": len(bps),
    }


def _tool_blueprint_get(args: dict) -> dict:
    from .blueprint_store import resolve_blueprint
    bp = resolve_blueprint(args["blueprint_id"])
    if not bp:
        return {"error": f"Blueprint '{args['blueprint_id']}' not found"}
    return bp.model_dump()


def _tool_storage_scope_list(_args: dict) -> dict:
    from .storage_scope import list_scopes

    scopes = list_scopes()
    return {"scopes": scopes, "count": len(scopes)}


def _tool_storage_scope_upsert(args: dict) -> dict:
    from .storage_scope import upsert_scope

    scope = upsert_scope(
        name=args.get("name", ""),
        roots=args.get("roots", []),
        approved_by=args.get("approved_by", "user"),
    )
    return {"stored": True, "scope": scope}


def _tool_storage_scope_delete(args: dict) -> dict:
    from .storage_scope import delete_scope

    name = str(args.get("name", "")).strip()
    if not name:
        return {"error": "name is required"}
    deleted = delete_scope(name)
    return {"deleted": bool(deleted), "name": name}


def _tool_storage_provision_container(args: dict) -> dict:
    """Atomic: create dir → scope upsert → blueprint_create (with mount) → optional request_container."""
    blueprint_id = str(args.get("blueprint_id") or "").strip()
    storage_host_path = str(args.get("storage_host_path") or "").strip()

    if not blueprint_id:
        return {"error": "blueprint_id is required"}
    if not storage_host_path:
        return {"error": "storage_host_path is required"}
    if not os.path.isabs(storage_host_path):
        return {"error": f"storage_host_path must be an absolute path, got: '{storage_host_path}'"}

    container_mount_path = str(args.get("container_mount_path") or "/app/data").strip()
    storage_mode = str(args.get("storage_mode") or "rw").strip().lower()
    if storage_mode not in ("ro", "rw"):
        return {"error": f"storage_mode must be 'ro' or 'rw', got: '{storage_mode}'"}

    scope_name = f"{blueprint_id}_scope"

    # Step 0: Ensure host directory exists with correct ownership.
    # Docker auto-creates missing bind-mount dirs as root:root which breaks non-root containers.
    dir_created = False
    dir_chowned = False
    if not os.path.exists(storage_host_path):
        try:
            os.makedirs(storage_host_path, mode=0o750, exist_ok=True)
            dir_created = True
            logger.info(f"[provision] Created storage dir: {storage_host_path} (mode=0o750)")
        except Exception as e:
            return {"error": f"Failed to create storage_host_path '{storage_host_path}': {e}"}

    # Optional chown so non-root container users can write.
    owner_uid = args.get("owner_uid")
    owner_gid = args.get("owner_gid")
    if owner_uid is not None and dir_created:
        try:
            uid = int(owner_uid)
            gid = int(owner_gid) if owner_gid is not None else uid
            os.chown(storage_host_path, uid, gid)
            dir_chowned = True
            logger.info(f"[provision] chown {storage_host_path} → uid={uid} gid={gid}")
        except Exception as e:
            logger.warning(f"[provision] chown failed for '{storage_host_path}': {e}")

    # Step 1: Register storage scope
    try:
        scope_result = _tool_storage_scope_upsert({
            "name": scope_name,
            "roots": [{"path": storage_host_path, "mode": storage_mode}],
            "approved_by": "user",
        })
    except Exception as e:
        return {"error": f"storage_scope_upsert failed: {e}"}

    # Step 2: Create blueprint with mount
    bp_args = {
        "id": blueprint_id,
        "image": args["image"],
        "name": args["name"],
        "description": args.get("description", ""),
        "memory_limit": args.get("memory_limit", "512m"),
        "cpu_limit": args.get("cpu_limit", "0.5"),
        "network": args.get("network", "internal"),
        "environment": args.get("environment", {}),
        "allowed_exec": args.get("allowed_exec", []),
        "tags": args.get("tags", []),
        "storage_scope": scope_name,
        "mounts": [
            {"host": storage_host_path, "container": container_mount_path, "mode": storage_mode, "type": "bind"}
        ],
    }
    bp_result = _tool_blueprint_create(bp_args)
    if "error" in bp_result:
        return {
            "error": f"blueprint_create failed: {bp_result['error']}",
            "scope_created": scope_result.get("stored", False),
            "scope_name": scope_name,
        }

    # Step 3: Optionally start the container
    auto_start = args.get("auto_start", True)
    container_result: dict = {}
    if auto_start:
        container_result = _tool_request_container({"blueprint_id": blueprint_id})

    result: dict = {
        "provisioned": True,
        "blueprint_id": blueprint_id,
        "scope_name": scope_name,
        "storage_host_path": storage_host_path,
        "container_mount_path": container_mount_path,
        "storage_mode": storage_mode,
        "blueprint_created": bp_result.get("created", False),
        "steps": {
            "0_dir_create": "created" if dir_created else "already_existed",
            "1_scope_upsert": "ok",
            "2_blueprint_create": "ok",
        },
    }
    if dir_chowned:
        result["steps"]["0_dir_create"] = "created+chowned"
    if auto_start:
        started = "container_id" in container_result and not container_result.get("error")
        result["steps"]["3_container_start"] = "ok" if started else "failed"
        if started:
            result["container_id"] = container_result.get("container_id")
        else:
            result["start_error"] = container_result.get("error", "unknown")
    return result


def _tool_snapshot_list(args: dict) -> dict:
    from .volumes import list_snapshots
    snaps = list_snapshots(volume_name=args.get("volume_name"))
    return {"snapshots": snaps, "count": len(snaps)}


def _tool_snapshot_restore(args: dict) -> dict:
    from .volumes import restore_snapshot
    vol = restore_snapshot(args["filename"], args.get("target_volume"))
    if not vol:
        return {"error": "Restore failed"}
    return {
        "restored": True,
        "volume": vol,
        "hint": "Nutze request_container(blueprint_id=..., resume_volume='" + vol + "') um den Workspace fortzusetzen.",
    }


def _tool_optimize(args: dict) -> dict:
    """Adjust resource limits on a running container via Docker update."""
    from .engine import get_client, _parse_memory
    client = get_client()
    container_id = args["container_id"]

    try:
        container = client.containers.get(container_id)
        update_kwargs = {}

        if args.get("cpu_limit"):
            cpu = float(args["cpu_limit"])
            update_kwargs["cpu_quota"] = int(cpu * 100000)
            update_kwargs["cpu_period"] = 100000

        if args.get("memory_limit"):
            mem = _parse_memory(args["memory_limit"])
            update_kwargs["mem_limit"] = mem
            update_kwargs["memswap_limit"] = mem * 2

        if not update_kwargs:
            return {"error": "No changes specified. Provide cpu_limit and/or memory_limit."}

        container.update(**update_kwargs)

        return {
            "optimized": True,
            "container_id": container_id,
            "new_limits": {
                "cpu": args.get("cpu_limit", "unchanged"),
                "memory": args.get("memory_limit", "unchanged"),
            },
        }
    except Exception as e:
        return {"error": f"Optimize failed: {str(e)}"}


# ── TRION Home Container ──────────────────────────────────

def _home_runtime_config() -> Dict[str, str]:
    try:
        from utils.trion_home_identity import load_home_identity

        identity = load_home_identity(create_if_missing=True)
    except Exception:
        identity = {}
    blueprint_id = str(identity.get("container_id") or "trion-home").strip() or "trion-home"
    container_path = (
        str(os.environ.get("TRION_HOME_CONTAINER_PATH", "")).strip()
        or str(identity.get("container_home_path") or "").strip()
        or "/home/trion"
    )
    volume_name = str(os.environ.get("TRION_HOME_VOLUME", "trion_home_data")).strip() or "trion_home_data"
    return {
        "blueprint_id": blueprint_id,
        "volume_name": volume_name,
        "container_path": container_path,
    }


_HOME_RUNTIME = _home_runtime_config()
TRION_HOME_BLUEPRINT_ID = _HOME_RUNTIME["blueprint_id"]
TRION_HOME_VOLUME = _HOME_RUNTIME["volume_name"]
TRION_HOME_PATH = _HOME_RUNTIME["container_path"]

_trion_home_container_id = None  # Cached container ID


def _ensure_trion_home() -> str:
    """Ensure TRION home container is running. Returns container_id."""
    global _trion_home_container_id
    from .engine import start_container, list_containers
    from .blueprint_store import get_blueprint, create_blueprint
    from .models import Blueprint, ResourceLimits, MountDef, NetworkMode
    
    # Check if already running
    running = list_containers()
    for c in running:
        if c.blueprint_id == TRION_HOME_BLUEPRINT_ID:
            _trion_home_container_id = c.container_id
            return c.container_id
    
    # Ensure blueprint exists
    bp = get_blueprint(TRION_HOME_BLUEPRINT_ID)
    if not bp:
        logger.info("[TRION Home] Creating trion-home blueprint...")
        bp = Blueprint(
            id=TRION_HOME_BLUEPRINT_ID,
            name="TRION Home Workspace",
            description="TRIONs persistenter Arbeitsbereich für Notizen, Projekte und Experimente",
            image="python:3.12-slim",
            resources=ResourceLimits(memory_limit="512m", cpu_limit="0.5", timeout_seconds=0),  # No timeout
            mounts=[MountDef(host=TRION_HOME_VOLUME, container=TRION_HOME_PATH, type="volume")],
            network=NetworkMode.INTERNAL,
            tags=["system", "persistent", "home"],
            icon="🏠",
        )
        create_blueprint(bp)
    
    # Start container
    instance = start_container(
        blueprint_id=TRION_HOME_BLUEPRINT_ID,
        resume_volume=TRION_HOME_VOLUME,
    )
    _trion_home_container_id = instance.container_id
    
    # Create home directory structure
    from .engine import exec_in_container
    exec_in_container(_trion_home_container_id, f"mkdir -p {TRION_HOME_PATH}/notes {TRION_HOME_PATH}/projects {TRION_HOME_PATH}/scripts {TRION_HOME_PATH}/.config", timeout=10)
    
    return _trion_home_container_id


# ── New Tool Implementations ──────────────────────────────

def _tool_blueprint_create(args: dict) -> dict:
    """Create a new blueprint from a trusted Docker image."""
    from .blueprint_store import create_blueprint, get_blueprint
    from .models import Blueprint, ResourceLimits, MountDef, NetworkMode
    from .trust import evaluate_blueprint_trust, is_trusted_image

    image = args["image"]
    blueprint_id = args["id"]

    # Security: Validate trusted image
    if not is_trusted_image(image):
        return {
            "error": f"Image '{image}' is not trusted. Only official Docker images are allowed.",
            "allowed_prefixes": [
                "ubuntu:",
                "python:",
                "node:",
                "postgres:",
                "mongo:",
                "redis:",
                "nginx:",
                "alpine:",
                "josh5/steam-headless",
            ],
        }

    # Check if blueprint already exists
    existing = get_blueprint(blueprint_id)
    if existing:
        return {"error": f"Blueprint '{blueprint_id}' already exists. Use a different ID."}

    # Parse mounts from args (new parameter: host→container mappings)
    raw_mounts = args.get("mounts") or []
    mounts: list = []
    for m in raw_mounts:
        if not isinstance(m, dict):
            continue
        host = str(m.get("host") or "").strip()
        container = str(m.get("container") or "").strip()
        if not host or not container:
            return {"error": f"Invalid mount entry {m!r}: 'host' and 'container' are required."}
        mode = str(m.get("mode") or "rw").strip().lower()
        if mode not in ("ro", "rw"):
            return {"error": f"Invalid mount mode '{mode}' for host='{host}'. Use 'ro' or 'rw'."}
        mtype = str(m.get("type") or "bind").strip().lower()
        if mtype not in ("bind", "volume"):
            return {"error": f"Invalid mount type '{mtype}' for host='{host}'. Use 'bind' or 'volume'."}
        mounts.append(MountDef(host=host, container=container, mode=mode, type=mtype))

    # Build blueprint
    bp = Blueprint(
        id=blueprint_id,
        name=args["name"],
        description=args.get("description", ""),
        image=image,
        resources=ResourceLimits(
            memory_limit=args.get("memory_limit", "512m"),
            cpu_limit=args.get("cpu_limit", "0.5"),
        ),
        network=NetworkMode(args.get("network", "internal")),
        ports=args.get("ports", []),
        runtime=args.get("runtime", ""),
        devices=args.get("devices", []),
        environment=args.get("environment", {}),
        storage_scope=args.get("storage_scope", ""),
        mounts=mounts,
        healthcheck=args.get("healthcheck", {}),
        cap_add=args.get("cap_add", []),
        shm_size=args.get("shm_size", ""),
        tags=args.get("tags", []),
        allowed_exec=args.get("allowed_exec", []),
        icon="📦",
    )

    # Determine trust level via trust.py (single source of truth)
    try:
        _trust_decision = evaluate_blueprint_trust(bp)
        _trust_level = _trust_decision["level"] if _trust_decision["level"] in ("verified", "unverified") else "unverified"
    except Exception:
        _trust_level = "unverified"

    created = create_blueprint(bp)
    logger.info(f"[Blueprint] Created: {created.id} (image={image}, trust={_trust_level})")

    # Sync new blueprint to graph immediately
    try:
        from .blueprint_store import sync_blueprint_to_graph
        sync_blueprint_to_graph(created, trust_level=_trust_level)
    except Exception as _e:
        logger.warning(f"[Blueprint] Graph sync failed for {created.id}: {_e}")

    return {
        "created": True,
        "blueprint_id": created.id,
        "name": created.name,
        "image": created.image,
        "hint": f"Starte den Container mit: request_container(blueprint_id='{created.id}')",
    }


def _tool_home_write(args: dict) -> dict:
    """Write a file to TRION's home container."""
    from .engine import exec_in_container
    import base64
    
    # Validate required parameters
    if "path" not in args:
        return {"error": "Missing required parameter 'path'. Example: 'notes/todo.md'"}
    if "content" not in args:
        return {"error": "Missing required parameter 'content'. Provide the text to write."}
    
    container_id = _ensure_trion_home()
    path = args["path"].lstrip("/")  # Relative to home
    content = args["content"]
    
    # Encode content to base64 to handle special characters
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    
    # Create parent directories and write file
    full_path = f"{TRION_HOME_PATH}/{path}"
    # Get parent directory
    if "/" in path:
        parent_dir = f"{TRION_HOME_PATH}/{'/'.join(path.split('/')[:-1])}"
    else:
        parent_dir = TRION_HOME_PATH
    
    exit_code, output = exec_in_container(
        container_id,
        f"sh -c \"mkdir -p '{parent_dir}' && echo '{encoded}' | base64 -d > '{full_path}'\"",
        timeout=10
    )
    
    if exit_code != 0:
        return {"error": f"Write failed: {output}"}
    
    return {
        "written": True,
        "path": path,
        "size": len(content),
        "container_id": container_id,
    }


def _tool_home_read(args: dict) -> dict:
    """Read a file from TRION's home container."""
    from .engine import exec_in_container
    
    # Validate required parameters
    if "path" not in args:
        return {"error": "Missing required parameter 'path'. Example: 'notes/todo.md'"}
    
    container_id = _ensure_trion_home()
    path = args["path"].lstrip("/")
    full_path = f"{TRION_HOME_PATH}/{path}"
    
    exit_code, output = exec_in_container(
        container_id,
        f"sh -c \"cat '{full_path}'\"",
        timeout=10
    )
    
    if exit_code != 0:
        return {"error": f"Read failed: {output}", "path": path}
    
    return {
        "content": output,
        "path": path,
        "container_id": container_id,
    }


def _tool_home_list(args: dict) -> dict:
    """List contents of a directory in TRION's home container."""
    from .engine import exec_in_container
    
    container_id = _ensure_trion_home()
    path = args.get("path", ".").lstrip("/")
    full_path = f"{TRION_HOME_PATH}/{path}" if path and path != "." else TRION_HOME_PATH
    
    exit_code, output = exec_in_container(
        container_id,
        f"sh -c \"ls -la '{full_path}'\"",
        timeout=10
    )
    
    if exit_code != 0:
        return {"error": f"List failed: {output}", "path": path}
    
    return {
        "listing": output,
        "path": path,
        "container_id": container_id,
    }


# ── Discovery Tools ───────────────────────────────────────

def _tool_container_list(args: dict) -> dict:
    """List all TRION-managed containers with status + blueprint info."""
    from .engine import list_containers
    status_filter = args.get("status_filter", "all")

    instances = list_containers()
    containers = []
    for inst in instances:
        if status_filter == "running" and inst.status.value != "running":
            continue
        if status_filter == "stopped" and inst.status.value not in ("stopped", "exited"):
            continue
        containers.append({
            "container_id": inst.container_id,
            "name": inst.name,
            "blueprint_id": inst.blueprint_id,
            "status": inst.status.value,
            "started_at": inst.started_at or "",
            "ttl_remaining_seconds": int(inst.ttl_remaining) if inst.ttl_remaining else None,
            "volume": inst.volume_name or "",
        })

    return {
        "containers": containers,
        "count": len(containers),
        "filter": status_filter,
    }


def _tool_container_inspect(args: dict) -> dict:
    """Get detailed info about a specific container."""
    from .engine import inspect_container
    container_id = args.get("container_id", "").strip()
    if not container_id:
        return {"error": "Missing required parameter 'container_id'"}
    return inspect_container(container_id)


# ── Autonomy Cron Tools ──────────────────────────────────

def _run_async_sync(coro):
    """
    Run async scheduler methods safely from this synchronous tool layer.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_holder = {"value": None, "error": None}

    def _runner():
        try:
            result_holder["value"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - defensive
            result_holder["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if result_holder["error"] is not None:
        raise result_holder["error"]
    return result_holder["value"]


def _get_autonomy_cron_scheduler():
    try:
        from core.autonomy.cron_runtime import get_scheduler

        return get_scheduler()
    except Exception:
        return None


def _tool_autonomy_cron_status(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    return _run_async_sync(scheduler.get_status())


def _tool_autonomy_cron_list_jobs(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    jobs = _run_async_sync(scheduler.list_jobs())
    return {"jobs": jobs, "count": len(jobs)}


def _tool_autonomy_cron_validate(args: dict) -> dict:
    cron_expr = str(args.get("cron", "")).strip()
    if not cron_expr:
        return {"valid": False, "error": "missing cron"}
    try:
        from core.autonomy.cron_scheduler import validate_cron_expression

        validated = validate_cron_expression(cron_expr)
        return {"valid": True, **validated}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def _reference_links_rows_for_category(
    category: str,
    *,
    include_disabled: bool = False,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    from utils.settings import settings as runtime_settings

    categories = {"cronjobs", "skills", "blueprints"}
    requested = str(category or "").strip().lower()
    if requested not in categories:
        return []

    raw = runtime_settings.get("TRION_REFERENCE_LINK_COLLECTIONS", {})
    source = raw if isinstance(raw, dict) else {}
    rows = source.get(requested, [])
    if not isinstance(rows, list):
        return []

    out: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for entry in rows:
        item = entry if isinstance(entry, dict) else {}
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        if not name or not url:
            continue
        enabled = bool(item.get("enabled", True))
        if not include_disabled and not enabled:
            continue
        key = url.lower()
        if key in seen_urls:
            continue
        seen_urls.add(key)
        out.append(
            {
                "name": name[:120],
                "url": url[:500],
                "description": str(item.get("description", "")).strip()[:300],
                "enabled": enabled,
                "read_only": True,
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return out


def _tool_autonomy_cron_create_job(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    payload = {
        "name": args.get("name"),
        "objective": args.get("objective"),
        "conversation_id": args.get("conversation_id") or args.get("session_id") or "",
        "cron": args.get("cron"),
        "schedule_mode": args.get("schedule_mode", "recurring"),
        "run_at": args.get("run_at", ""),
        "timezone": args.get("timezone", "UTC"),
        "max_loops": args.get("max_loops", 10),
        "created_by": args.get("created_by", "user"),
        "user_approved": bool(args.get("user_approved", False)),
        "enabled": bool(args.get("enabled", True)),
    }
    auto_refs: List[Dict[str, Any]] = []
    created_by = str(payload.get("created_by", "user")).strip().lower()
    if created_by == "trion":
        auto_refs = _reference_links_rows_for_category("cronjobs", include_disabled=False, limit=8)
        if auto_refs:
            payload["reference_links"] = auto_refs
            payload["reference_source"] = "settings:cronjobs:auto"
    try:
        created = _run_async_sync(scheduler.create_job(payload))
        if isinstance(created, dict):
            created["reference_links_used"] = {
                "category": "cronjobs",
                "count": len(auto_refs),
                "source": "settings:cronjobs:auto" if auto_refs else "",
            }
        return created
    except Exception as exc:
        error_code = getattr(exc, "error_code", "")
        details = getattr(exc, "details", None)
        if error_code:
            return {"error": str(exc), "error_code": error_code, "details": details or {}}
        return {"error": str(exc)}


def _tool_autonomy_cron_update_job(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    cron_job_id = str(args.get("cron_job_id", "")).strip()
    if not cron_job_id:
        return {"error": "missing cron_job_id"}
    payload = {
        key: args[key]
        for key in [
            "name",
            "objective",
            "conversation_id",
            "cron",
            "schedule_mode",
            "run_at",
            "timezone",
            "max_loops",
            "created_by",
            "user_approved",
            "enabled",
        ]
        if key in args
    }
    try:
        updated = _run_async_sync(scheduler.update_job(cron_job_id, payload))
    except Exception as exc:
        error_code = getattr(exc, "error_code", "")
        details = getattr(exc, "details", None)
        if error_code:
            return {"error": str(exc), "error_code": error_code, "details": details or {}}
        return {"error": str(exc)}
    if not updated:
        return {"error": "cron_job_not_found", "cron_job_id": cron_job_id}
    return updated


def _tool_autonomy_cron_pause_job(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    cron_job_id = str(args.get("cron_job_id", "")).strip()
    if not cron_job_id:
        return {"error": "missing cron_job_id"}
    paused = _run_async_sync(scheduler.pause_job(cron_job_id))
    if not paused:
        return {"error": "cron_job_not_found", "cron_job_id": cron_job_id}
    return paused


def _tool_autonomy_cron_resume_job(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    cron_job_id = str(args.get("cron_job_id", "")).strip()
    if not cron_job_id:
        return {"error": "missing cron_job_id"}
    resumed = _run_async_sync(scheduler.resume_job(cron_job_id))
    if not resumed:
        return {"error": "cron_job_not_found", "cron_job_id": cron_job_id}
    return resumed


def _tool_autonomy_cron_run_now(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    cron_job_id = str(args.get("cron_job_id", "")).strip()
    if not cron_job_id:
        return {"error": "missing cron_job_id"}
    try:
        scheduled = _run_async_sync(scheduler.run_now(cron_job_id, reason="tool"))
    except Exception as exc:
        error_code = getattr(exc, "error_code", "")
        details = getattr(exc, "details", None)
        if error_code:
            return {"error": str(exc), "error_code": error_code, "details": details or {}}
        return {"error": str(exc)}
    if not scheduled:
        return {"error": "cron_job_not_found", "cron_job_id": cron_job_id}
    return scheduled


def _tool_autonomy_cron_delete_job(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    cron_job_id = str(args.get("cron_job_id", "")).strip()
    if not cron_job_id:
        return {"error": "missing cron_job_id"}
    deleted = _run_async_sync(scheduler.delete_job(cron_job_id))
    return {"deleted": bool(deleted), "cron_job_id": cron_job_id}


def _tool_autonomy_cron_queue(args: dict) -> dict:
    scheduler = _get_autonomy_cron_scheduler()
    if not scheduler:
        return {"error": "autonomy_cron_unavailable"}
    return _run_async_sync(scheduler.get_queue_snapshot())


def _tool_cron_reference_links_list(args: dict) -> dict:
    categories = ("cronjobs", "skills", "blueprints")
    requested_category = str(args.get("category", "")).strip().lower()
    include_disabled = bool(args.get("include_disabled", False))
    limit = int(args.get("limit", 50) or 50)
    limit = max(1, min(100, limit))

    if requested_category and requested_category not in categories:
        return {"error": "invalid_category", "allowed_categories": list(categories)}

    if requested_category:
        links = _reference_links_rows_for_category(
            requested_category,
            include_disabled=include_disabled,
            limit=limit,
        )
        return {
            "category": requested_category,
            "count": len(links),
            "links": links,
            "mode": "read_only_for_trion",
            "include_disabled": include_disabled,
        }

    collections: Dict[str, List[Dict[str, Any]]] = {}
    for category in categories:
        collections[category] = _reference_links_rows_for_category(
            category,
            include_disabled=include_disabled,
            limit=limit,
        )

    return {
        "categories": list(categories),
        "collections": collections,
        "mode": "read_only_for_trion",
        "include_disabled": include_disabled,
    }


# ── Registration Helper ───────────────────────────────────

def get_tool_definitions(include_fast_lane: bool = True) -> List[Dict]:
    """Return tool definitions for MCP registration."""
    if include_fast_lane:
        return TOOL_DEFINITIONS
    return [t for t in TOOL_DEFINITIONS if t["name"] not in FAST_LANE_TOOL_NAMES]


def get_system_prompt() -> str:
    """Return the system prompt that should be injected when containers are active."""
    return CONTAINER_SYSTEM_PROMPT
