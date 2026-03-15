"""
E2E-Tests für alle Fixes vom 2026-03-15.

Anforderungen:
  - jarvis-admin-api läuft auf localhost:8200
  - Container mounten /app als Volume (live-Code)

Ausführen:
  pytest tests/e2e/test_todays_fixes_e2e.py -v

Abgedeckte Fixes:
  Fix #3  INV-15: Control Layer Blueprint-Gate als Routing-Signal
  Fix #4  Cron Scheduler: Keyword-Panik + max_loops 12→50
  Fix #5  Tool-Amnesie: ThinkingLayer AUSNAHME-Regel für needs_memory
  Fix #6  Fragile 4-Step-Chain: blueprint_create mounts + storage_provision_container
  Fix #7  dry_run-Falle: mount_utils verhindert Docker root:root-Auto-Create
  Fix #8  Zonen vs. Scopes: blueprint_create erkennt zone-Namen früh
"""

import os
import subprocess
import json
import tempfile
import requests
import pytest
from types import SimpleNamespace

# ── Config ──────────────────────────────────────────────────────────

ADMIN_API = os.environ.get("TRION_ADMIN_API_URL", "http://localhost:8200")
CONTAINER  = "jarvis-admin-api"
APP_PATH   = "/app"


def _docker_run(script: str) -> dict:
    """Run a Python snippet inside the admin-api container. Returns parsed JSON."""
    cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    raw = result.stdout.strip()
    # Find last JSON-looking line
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            return json.loads(line)
    raise RuntimeError(f"No JSON output.\nstdout:\n{raw}\nstderr:\n{result.stderr[:500]}")


def _docker_eval(snippet: str) -> str:
    """Run snippet, return last non-log line."""
    cmd = ["docker", "exec", CONTAINER, "python3", "-u", "-c",
           f"import sys; sys.path.insert(0,'{APP_PATH}')\n{snippet}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    lines = [l for l in r.stdout.splitlines() if not l.startswith("[20")]
    return "\n".join(lines).strip()


@pytest.fixture(scope="session", autouse=True)
def check_services():
    """Skip all tests if admin-api is not reachable."""
    try:
        r = requests.get(f"{ADMIN_API}/health", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        pytest.skip(f"admin-api not reachable at {ADMIN_API}: {e}")


# ══════════════════════════════════════════════════════════════════════
# Fix #3 — INV-15: Control Layer Blueprint-Gate als Routing-Signal
# ══════════════════════════════════════════════════════════════════════

class TestFix3BlueprintGate:
    """CONTROL_PROMPT muss blueprint_gate_blocked als Routing-Signal behandeln, nicht als Safety-Block."""

    def test_control_prompt_contains_blueprint_gate_rule(self):
        out = _docker_eval(
            "from core.layers.control import CONTROL_PROMPT\n"
            "print('PRESENT' if 'blueprint_gate_blocked' in CONTROL_PROMPT else 'MISSING')"
        )
        assert out == "PRESENT", "CONTROL_PROMPT must contain blueprint_gate_blocked rule"

    def test_control_prompt_routing_not_safety_block(self):
        out = _docker_eval(
            "from core.layers.control import CONTROL_PROMPT\n"
            "print('PRESENT' if 'ROUTING-SIGNAL' in CONTROL_PROMPT else 'MISSING')"
        )
        assert out == "PRESENT", "CONTROL_PROMPT must describe blueprint_gate_blocked as ROUTING-SIGNAL"

    def test_stabilize_overrides_approved_false_when_gate_blocked(self):
        """_stabilize_verification_result muss approved=true setzen wenn blueprint_gate_blocked=true."""
        out = _docker_eval(
            "from core.layers.control import ControlLayer\n"
            "cl = ControlLayer.__new__(ControlLayer)\n"
            "plan = {'blueprint_gate_blocked': True}\n"
            "decision = {'approved': False, 'hard_block': True, 'decision_class': 'block', 'final_instruction': ''}\n"
            "result = cl._stabilize_verification_result(decision, plan)\n"
            "print(result.get('approved'), result.get('hard_block'))"
        )
        assert out == "True False", f"Expected 'True False', got: {out!r}"


# ══════════════════════════════════════════════════════════════════════
# Fix #4 — Cron Scheduler: Keyword-Panik + max_loops
# ══════════════════════════════════════════════════════════════════════

class TestFix4CronScheduler:
    """Context-approvierte Objectives nicht blockieren; immer-riskante blockieren; max_loops=50."""

    def _run_policy(self, objective: str, expected_block: bool):
        script = (
            f"import sys; sys.path.insert(0,'/app')\n"
            f"from core.autonomy.cron_scheduler import AutonomyCronScheduler, CronPolicyError\n"
            f"s = object.__new__(AutonomyCronScheduler)\n"
            f"s._trion_require_approval_for_risky = True\n"
            f"s._trion_safe_mode = True\n"
            f"s._trion_min_interval_s = 60\n"
            f"s._trion_max_loops = 50\n"
            f"norm = {{'name':'t','objective':{objective!r},'cron':'0 * * * *','created_by':'trion','user_approved':False}}\n"
            f"try:\n"
            f"    s._enforce_trion_policy_locked(norm, {{}}, creating=True, min_interval_s=60)\n"
            f"    print('PASSED')\n"
            f"except CronPolicyError as e:\n"
            f"    print('BLOCKED:' + e.error_code)\n"
        )
        cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = next((l for l in r.stdout.splitlines() if l.startswith(("PASSED", "BLOCKED"))), "ERROR")
        blocked = out.startswith("BLOCKED")
        assert blocked == expected_block, (
            f"objective={objective!r}: expected_block={expected_block}, got={out!r}"
        )

    def test_always_risky_drop_blocked(self):
        self._run_policy("drop all user tables", expected_block=True)

    def test_always_risky_wipe_blocked(self):
        self._run_policy("wipe database completely", expected_block=True)

    def test_always_risky_steal_credentials_blocked(self):
        self._run_policy("steal credentials from vault", expected_block=True)

    def test_context_approved_restart_health(self):
        self._run_policy("restart failed service after health check", expected_block=False)

    def test_context_approved_delete_cleanup(self):
        self._run_policy("delete old log files from cleanup job", expected_block=False)

    def test_context_approved_docker_status(self):
        self._run_policy("docker container cleanup and status report", expected_block=False)

    def test_max_loops_is_50(self):
        out = _docker_eval(
            "from config import get_autonomy_cron_trion_max_loops\n"
            "print(get_autonomy_cron_trion_max_loops())"
        )
        assert out.strip() == "50", f"max_loops should be 50, got: {out!r}"


# ══════════════════════════════════════════════════════════════════════
# Fix #5 — Tool-Amnesie: ThinkingLayer AUSNAHME-Regel
# ══════════════════════════════════════════════════════════════════════

class TestFix5ToolAmnesia:
    """THINKING_PROMPT must contain the AUSNAHME rule for context-sensitive tool requests."""

    @pytest.fixture(scope="class")
    def prompt(self):
        out = _docker_eval(
            "from core.layers.thinking import THINKING_PROMPT\n"
            "import json; print(json.dumps(THINKING_PROMPT))"
        )
        return json.loads(out)

    def test_ausnahme_rule_present(self, prompt):
        assert "AUSNAHME" in prompt

    def test_needs_memory_true_allowed(self, prompt):
        assert "needs_memory: true" in prompt

    def test_needs_memory_false_default_preserved(self, prompt):
        assert "needs_memory: false" in prompt

    def test_gestern_example_present(self, prompt):
        assert "gestern" in prompt

    def test_memory_keys_not_empty_rule(self, prompt):
        assert "memory_keys NICHT leer" in prompt

    def test_runtime_haertung_section_present(self, prompt):
        assert "Runtime-Härtung" in prompt


# ══════════════════════════════════════════════════════════════════════
# Fix #6 — Fragile 4-Step-Chain: blueprint_create mounts + storage_provision_container
# ══════════════════════════════════════════════════════════════════════

class TestFix6StorageProvision:
    """storage_provision_container und blueprint_create.mounts müssen registriert und funktional sein."""

    @pytest.fixture(scope="class")
    def tool_defs(self):
        out = _docker_eval(
            "import json\n"
            "from container_commander.mcp_tools import TOOL_DEFINITIONS\n"
            "print(json.dumps([t['name'] for t in TOOL_DEFINITIONS]))"
        )
        return json.loads(out)

    def test_storage_provision_container_registered(self, tool_defs):
        assert "storage_provision_container" in tool_defs

    def test_blueprint_create_has_mounts_parameter(self):
        out = _docker_eval(
            "import json\n"
            "from container_commander.mcp_tools import TOOL_DEFINITIONS\n"
            "bp = next(t for t in TOOL_DEFINITIONS if t['name'] == 'blueprint_create')\n"
            "print('YES' if 'mounts' in bp.get('inputSchema', {}).get('properties', {}) else 'NO')"
        )
        assert out == "YES"

    def test_provision_container_call_returns_provisioned(self):
        """storage_provision_container mit existierendem tmp-Pfad muss provisioned=True zurückgeben."""
        script = (
            "import sys, os, json, tempfile; sys.path.insert(0, '/app')\n"
            "from container_commander.mcp_tools import call_tool\n"
            "from unittest.mock import patch\n"
            "tmpd = tempfile.mkdtemp()\n"
            "import container_commander.mcp_tools as t\n"
            "with patch.object(t, '_tool_storage_scope_upsert', lambda a: {'stored': True}), \\\n"
            "     patch.object(t, '_tool_blueprint_create', lambda a: {'created': True, 'blueprint_id': a['id']}), \\\n"
            "     patch.object(t, '_tool_request_container', lambda a: {'container_id': 'abc123'}):\n"
            "    r = t._tool_storage_provision_container({\n"
            "        'blueprint_id': 'e2e-prov-test',\n"
            "        'image': 'python:3.12-slim',\n"
            "        'name': 'E2E Test',\n"
            "        'storage_host_path': tmpd,\n"
            "    })\n"
            "print(json.dumps(r))\n"
        )
        cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        assert out, f"No JSON output:\n{r.stdout}\n{r.stderr[:300]}"
        result = json.loads(out)
        assert result.get("provisioned") is True
        assert result["steps"]["1_scope_upsert"] == "ok"
        assert result["steps"]["2_blueprint_create"] == "ok"
        assert result["steps"]["3_container_start"] == "ok"


# ══════════════════════════════════════════════════════════════════════
# Fix #7 — dry_run-Falle: mount_utils verhindert Docker root:root
# ══════════════════════════════════════════════════════════════════════

class TestFix7MountUtils:
    """ensure_bind_mount_host_dirs muss fehlende bind-mount Verzeichnisse vorab anlegen."""

    def test_creates_missing_bind_dir(self):
        script = (
            "import sys, os, json, tempfile; sys.path.insert(0, '/app')\n"
            "from container_commander.mount_utils import ensure_bind_mount_host_dirs\n"
            "from types import SimpleNamespace\n"
            "tmpd = tempfile.mkdtemp()\n"
            "new_dir = os.path.join(tmpd, 'missing-bind')\n"
            "mounts = [SimpleNamespace(host=new_dir, container='/app/data', mode='rw', type='bind')]\n"
            "ensure_bind_mount_host_dirs(mounts)\n"
            "print(json.dumps({'exists': os.path.exists(new_dir)}))\n"
        )
        cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        assert out, f"No output:\n{r.stdout}"
        result = json.loads(out)
        assert result["exists"] is True

    def test_skips_volume_type(self):
        script = (
            "import sys, os, json, tempfile; sys.path.insert(0, '/app')\n"
            "from container_commander.mount_utils import ensure_bind_mount_host_dirs\n"
            "from types import SimpleNamespace\n"
            "tmpd = tempfile.mkdtemp()\n"
            "phantom = os.path.join(tmpd, 'phantom-vol')\n"
            "mounts = [SimpleNamespace(host=phantom, container='/data', mode='rw', type='volume')]\n"
            "ensure_bind_mount_host_dirs(mounts)\n"
            "print(json.dumps({'exists': os.path.exists(phantom)}))\n"
        )
        cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        assert out
        assert json.loads(out)["exists"] is False

    def test_provision_container_autocreates_dir(self):
        """storage_provision_container darf nicht fehlschlagen wenn Pfad fehlt — Verzeichnis wird erstellt."""
        script = (
            "import sys, os, json, tempfile; sys.path.insert(0, '/app')\n"
            "from container_commander.mcp_tools import _tool_storage_provision_container\n"
            "from unittest.mock import patch\n"
            "import container_commander.mcp_tools as t\n"
            "tmpd = tempfile.mkdtemp()\n"
            "new_path = os.path.join(tmpd, 'autocreated-service')\n"
            "assert not os.path.exists(new_path)\n"
            "with patch.object(t, '_tool_storage_scope_upsert', lambda a: {'stored': True}), \\\n"
            "     patch.object(t, '_tool_blueprint_create', lambda a: {'created': True, 'blueprint_id': a['id']}), \\\n"
            "     patch.object(t, '_tool_request_container', lambda a: {'container_id': 'c1'}):\n"
            "    r = _tool_storage_provision_container({\n"
            "        'blueprint_id': 'autocreate-test',\n"
            "        'image': 'python:3.12-slim',\n"
            "        'name': 'AutoCreate',\n"
            "        'storage_host_path': new_path,\n"
            "    })\n"
            "print(json.dumps({'provisioned': r.get('provisioned'), 'dir_created': os.path.exists(new_path), 'step0': r['steps']['0_dir_create']}))\n"
        )
        cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        assert out, f"No JSON:\n{r.stdout}\n{r.stderr[:300]}"
        result = json.loads(out)
        assert result["provisioned"] is True
        assert result["dir_created"] is True
        assert result["step0"] == "created"

    def test_engine_calls_mount_utils(self):
        """engine.py muss ensure_bind_mount_host_dirs aufrufen."""
        out = _docker_eval(
            "src = open('/app/container_commander/engine.py').read()\n"
            "print('OK' if 'ensure_bind_mount_host_dirs(bp.mounts)' in src and 'mount_utils' in src else 'FAIL')"
        )
        assert out == "OK"


# ══════════════════════════════════════════════════════════════════════
# Fix #8 — Zonen vs. Scopes: blueprint_create Zone-Name-Guard
# ══════════════════════════════════════════════════════════════════════

class TestFix8ZoneScopeGuard:
    """blueprint_create muss Storage-Broker-Zonen-Namen früh abfangen."""

    ZONE_NAMES = ["managed_services", "backup", "system", "external", "docker_runtime", "unzoned"]

    @pytest.mark.parametrize("zone", ZONE_NAMES)
    def test_zone_name_as_scope_is_rejected(self, zone):
        script = (
            f"import sys, json; sys.path.insert(0, '/app')\n"
            f"from container_commander.mcp_tools import call_tool\n"
            f"r = call_tool('blueprint_create', {{'id': 'z1', 'image': 'python:3.12-slim', 'name': 'T', 'storage_scope': {zone!r}}})\n"
            f"print(json.dumps({{'has_error': 'error' in r, 'is_zone_error': 'Storage Broker zone' in r.get('error', ''), 'has_hint': 'hint' in r}}))\n"
        )
        cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        assert out, f"No JSON for zone={zone}:\n{r.stdout}"
        result = json.loads(out)
        assert result["has_error"], f"zone={zone}: should return error"
        assert result["is_zone_error"], f"zone={zone}: error must mention 'Storage Broker zone'"
        assert result["has_hint"], f"zone={zone}: must include remediation hint"

    def test_unregistered_scope_caught_early(self):
        script = (
            "import sys, json; sys.path.insert(0, '/app')\n"
            "from container_commander.mcp_tools import call_tool\n"
            "r = call_tool('blueprint_create', {'id': 'z2', 'image': 'python:3.12-slim', 'name': 'T', 'storage_scope': 'ghost-scope-xyz'})\n"
            "print(json.dumps({'has_error': 'error' in r, 'msg': r.get('error', '')[:80]}))\n"
        )
        cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        assert out
        result = json.loads(out)
        assert result["has_error"]
        assert "not been registered" in result["msg"] or "storage_scope_upsert" in result["msg"]

    def test_zone_guard_fires_before_scope_lookup(self):
        """Zone-Guard muss feuern BEVOR get_scope aufgerufen wird — kein redundanter DB-Lookup."""
        script = (
            "import sys, json; sys.path.insert(0, '/app')\n"
            "from container_commander.mcp_tools import call_tool\n"
            "from unittest.mock import patch, MagicMock\n"
            "import container_commander.storage_scope as ss\n"
            "calls = []\n"
            "orig = ss.get_scope\n"
            "ss.get_scope = lambda n: calls.append(n) or orig(n)\n"
            "r = call_tool('blueprint_create', {'id': 'z3', 'image': 'python:3.12-slim', 'name': 'T', 'storage_scope': 'managed_services'})\n"
            "ss.get_scope = orig\n"
            "print(json.dumps({'blocked': 'Storage Broker zone' in r.get('error', ''), 'scope_lookups': len(calls)}))\n"
        )
        cmd = ["docker", "exec", CONTAINER, "python3", "-c", script]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        assert out
        result = json.loads(out)
        assert result["blocked"], "Zone guard must fire"
        assert result["scope_lookups"] == 0, "get_scope must NOT be called when zone guard fires"

    def test_storage_scope_upsert_description_warns_zones(self):
        out = _docker_eval(
            "from container_commander.mcp_tools import TOOL_DEFINITIONS\n"
            "t = next(x for x in TOOL_DEFINITIONS if x['name'] == 'storage_scope_upsert')\n"
            "print('OK' if 'NOT a Storage Broker zone' in t['description'] else 'FAIL')"
        )
        assert out == "OK"

    def test_storage_provision_container_description_bridges_zones(self):
        out = _docker_eval(
            "from container_commander.mcp_tools import TOOL_DEFINITIONS\n"
            "t = next(x for x in TOOL_DEFINITIONS if x['name'] == 'storage_provision_container')\n"
            "desc = t['description']\n"
            "print('OK' if 'managed_services' in desc and 'scope' in desc.lower() else 'FAIL')"
        )
        assert out == "OK"
