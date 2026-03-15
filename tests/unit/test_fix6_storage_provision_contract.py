"""
Tests für Fix #6: Fragile 4-Step Tool Chain — Storage+Container Provisioning.

Fix A: blueprint_create now accepts a `mounts` parameter (host→container mappings).
Fix B: storage_provision_container composite tool replaces manual 3-step chain.
"""
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helpers ──────────────────────────────────────────────

def _mcp_src() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "container_commander/mcp_tools.py").read_text(encoding="utf-8")


# ── Fix A: blueprint_create mounts parameter ─────────────

def test_blueprint_create_tool_schema_has_mounts():
    """blueprint_create tool definition must expose a 'mounts' parameter."""
    src = _mcp_src()
    # Find the blueprint_create tool definition section
    assert '"name": "blueprint_create"' in src
    # The mounts property must appear in the schema
    assert '"mounts"' in src


def test_blueprint_create_implementation_parses_mounts():
    """_tool_blueprint_create must parse mounts arg and build MountDef objects."""
    src = _mcp_src()
    assert "raw_mounts = args.get(\"mounts\") or []" in src
    assert "MountDef(" in src
    assert "mounts=mounts" in src


def test_blueprint_create_mount_validation_rejects_missing_host():
    """_tool_blueprint_create must reject mounts with empty host or container."""
    src = _mcp_src()
    assert "'host' and 'container' are required" in src


def test_blueprint_create_mount_validation_rejects_bad_mode():
    """_tool_blueprint_create must reject mount mode values outside ro/rw."""
    src = _mcp_src()
    assert "Invalid mount mode" in src
    assert "'ro' or 'rw'" in src


def test_blueprint_create_mount_validation_rejects_bad_type():
    """_tool_blueprint_create must reject mount type values outside bind/volume."""
    src = _mcp_src()
    assert "Invalid mount type" in src
    assert "'bind' or 'volume'" in src


def test_blueprint_create_with_mounts_runtime(tmp_path):
    """Integration: blueprint_create with mounts builds correct Blueprint object."""
    # Stub out heavy imports
    created_blueprints = {}

    class FakeBlueprint:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.id = kw.get("id", "test-bp")
            self.name = kw.get("name", "Test")
            self.image = kw.get("image", "python:3.12-slim")
            self.mounts = kw.get("mounts", [])

    class FakeMountDef:
        def __init__(self, host, container, mode, type):
            self.host = host
            self.container = container
            self.mode = mode
            self.type = type

    class FakeResourceLimits:
        def __init__(self, **kw): pass

    class FakeNetworkMode(str):
        def __new__(cls, v): return str.__new__(cls, v)

    def fake_is_trusted(image): return True
    def fake_get_bp(bid): return None
    def fake_create(bp):
        created_blueprints[bp.id] = bp
        return bp
    def fake_evaluate(bp): return {"level": "verified"}
    def fake_sync(bp, trust_level=None): pass

    import container_commander.mcp_tools as tools

    with patch.dict("sys.modules", {
        "container_commander.blueprint_store": MagicMock(
            create_blueprint=fake_create,
            get_blueprint=fake_get_bp,
            sync_blueprint_to_graph=fake_sync,
        ),
        "container_commander.models": MagicMock(
            Blueprint=FakeBlueprint,
            MountDef=FakeMountDef,
            ResourceLimits=FakeResourceLimits,
            NetworkMode=FakeNetworkMode,
        ),
        "container_commander.trust": MagicMock(
            is_trusted_image=fake_is_trusted,
            evaluate_blueprint_trust=fake_evaluate,
        ),
    }):
        result = tools._tool_blueprint_create({
            "id": "test-service",
            "image": "python:3.12-slim",
            "name": "Test Service",
            "mounts": [{"host": "/data/test", "container": "/app/data", "mode": "rw", "type": "bind"}],
        })

    assert result.get("created") is True
    assert result.get("blueprint_id") == "test-service"
    bp = created_blueprints["test-service"]
    assert len(bp.mounts) == 1
    assert bp.mounts[0].host == "/data/test"
    assert bp.mounts[0].container == "/app/data"
    assert bp.mounts[0].mode == "rw"
    assert bp.mounts[0].type == "bind"


# ── Fix B: storage_provision_container composite tool ───

def test_storage_provision_container_tool_in_tools_list():
    """storage_provision_container must be registered in the TOOLS list."""
    src = _mcp_src()
    assert '"name": "storage_provision_container"' in src


def test_storage_provision_container_schema_required_fields():
    """storage_provision_container schema must require blueprint_id, image, name, storage_host_path."""
    src = _mcp_src()
    assert '"required": ["blueprint_id", "image", "name", "storage_host_path"]' in src


def test_storage_provision_container_schema_has_auto_start():
    src = _mcp_src()
    assert '"auto_start"' in src


def test_storage_provision_container_call_tool_dispatch():
    """call_tool must dispatch to _tool_storage_provision_container."""
    src = _mcp_src()
    assert 'tool_name == "storage_provision_container"' in src
    assert "_tool_storage_provision_container(arguments)" in src


def test_storage_provision_container_implementation_present():
    """_tool_storage_provision_container function must exist."""
    src = _mcp_src()
    assert "def _tool_storage_provision_container" in src


def test_storage_provision_container_validates_absolute_path():
    src = _mcp_src()
    assert "os.path.isabs(storage_host_path)" in src


def test_storage_provision_container_checks_path_exists():
    src = _mcp_src()
    assert "os.path.exists(storage_host_path)" in src
    assert "storage_create_service_dir" in src  # hint in error message


def test_storage_provision_container_uses_scope_upsert():
    src = _mcp_src()
    assert "_tool_storage_scope_upsert(" in src.split("def _tool_storage_provision_container")[1].split("def _tool_snapshot_list")[0]


def test_storage_provision_container_uses_blueprint_create():
    src = _mcp_src()
    impl = src.split("def _tool_storage_provision_container")[1].split("def _tool_snapshot_list")[0]
    assert "_tool_blueprint_create(" in impl


def test_storage_provision_container_scope_name_derived_from_blueprint_id():
    src = _mcp_src()
    impl = src.split("def _tool_storage_provision_container")[1].split("def _tool_snapshot_list")[0]
    assert 'f"{blueprint_id}_scope"' in impl


def test_storage_provision_container_auto_start_calls_request_container():
    src = _mcp_src()
    impl = src.split("def _tool_storage_provision_container")[1].split("def _tool_snapshot_list")[0]
    assert "_tool_request_container(" in impl
    assert "auto_start" in impl


def test_storage_provision_container_runtime_missing_path(tmp_path):
    """Returns error when storage_host_path does not exist."""
    import container_commander.mcp_tools as tools
    result = tools._tool_storage_provision_container({
        "blueprint_id": "test-bp",
        "image": "python:3.12-slim",
        "name": "Test",
        "storage_host_path": str(tmp_path / "nonexistent"),
    })
    assert "error" in result
    assert "does not exist" in result["error"]


def test_storage_provision_container_runtime_relative_path():
    """Returns error when storage_host_path is relative."""
    import container_commander.mcp_tools as tools
    result = tools._tool_storage_provision_container({
        "blueprint_id": "test-bp",
        "image": "python:3.12-slim",
        "name": "Test",
        "storage_host_path": "relative/path",
    })
    assert "error" in result
    assert "absolute" in result["error"]


def test_storage_provision_container_runtime_happy_path(tmp_path):
    """Full happy path: scope upsert + blueprint create + container start."""
    import container_commander.mcp_tools as tools

    host_path = str(tmp_path)

    scope_calls = []
    bp_calls = []
    rc_calls = []

    def fake_scope_upsert(args):
        scope_calls.append(args)
        return {"stored": True, "scope": {"name": args["name"]}}

    def fake_bp_create(args):
        bp_calls.append(args)
        return {"created": True, "blueprint_id": args["id"]}

    def fake_request_container(args):
        rc_calls.append(args)
        return {"container_id": "abc123", "status": "running"}

    with patch.object(tools, "_tool_storage_scope_upsert", fake_scope_upsert), \
         patch.object(tools, "_tool_blueprint_create", fake_bp_create), \
         patch.object(tools, "_tool_request_container", fake_request_container):

        result = tools._tool_storage_provision_container({
            "blueprint_id": "my-service",
            "image": "python:3.12-slim",
            "name": "My Service",
            "storage_host_path": host_path,
            "container_mount_path": "/app/data",
            "storage_mode": "rw",
            "auto_start": True,
        })

    assert result.get("provisioned") is True
    assert result.get("blueprint_id") == "my-service"
    assert result.get("scope_name") == "my-service_scope"
    assert result.get("container_id") == "abc123"
    assert result["steps"]["1_scope_upsert"] == "ok"
    assert result["steps"]["2_blueprint_create"] == "ok"
    assert result["steps"]["3_container_start"] == "ok"

    # Scope must be registered with host path
    assert scope_calls[0]["name"] == "my-service_scope"
    assert scope_calls[0]["roots"][0]["path"] == host_path

    # Blueprint must include mount + storage_scope
    bp_arg = bp_calls[0]
    assert bp_arg["storage_scope"] == "my-service_scope"
    assert len(bp_arg["mounts"]) == 1
    assert bp_arg["mounts"][0]["host"] == host_path
    assert bp_arg["mounts"][0]["container"] == "/app/data"

    # Container was started
    assert rc_calls[0]["blueprint_id"] == "my-service"


def test_storage_provision_container_runtime_no_auto_start(tmp_path):
    """auto_start=False: scope+blueprint created, no request_container call."""
    import container_commander.mcp_tools as tools

    host_path = str(tmp_path)
    rc_calls = []

    def fake_scope_upsert(args):
        return {"stored": True}

    def fake_bp_create(args):
        return {"created": True, "blueprint_id": args["id"]}

    def fake_request_container(args):
        rc_calls.append(args)  # should NOT be called

    with patch.object(tools, "_tool_storage_scope_upsert", fake_scope_upsert), \
         patch.object(tools, "_tool_blueprint_create", fake_bp_create), \
         patch.object(tools, "_tool_request_container", fake_request_container):

        result = tools._tool_storage_provision_container({
            "blueprint_id": "my-service",
            "image": "python:3.12-slim",
            "name": "My Service",
            "storage_host_path": host_path,
            "auto_start": False,
        })

    assert result.get("provisioned") is True
    assert "3_container_start" not in result["steps"]
    assert len(rc_calls) == 0


def test_storage_provision_container_propagates_blueprint_error(tmp_path):
    """If blueprint_create fails, return error with scope info for rollback awareness."""
    import container_commander.mcp_tools as tools

    host_path = str(tmp_path)

    with patch.object(tools, "_tool_storage_scope_upsert", lambda a: {"stored": True}), \
         patch.object(tools, "_tool_blueprint_create", lambda a: {"error": "Image not trusted"}):

        result = tools._tool_storage_provision_container({
            "blueprint_id": "bad-service",
            "image": "evil:latest",
            "name": "Bad",
            "storage_host_path": host_path,
        })

    assert "error" in result
    assert "blueprint_create failed" in result["error"]
    assert result.get("scope_created") is True  # scope was created before blueprint failed
    assert result.get("scope_name") == "bad-service_scope"
