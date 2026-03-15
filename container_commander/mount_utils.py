"""
Container Commander — Mount Utilities
======================================
Helpers for bind-mount host directory management.

Kept in a standalone module (no docker dep) so it can be unit-tested
without the full engine import chain.
"""
import logging
import os

logger = logging.getLogger(__name__)


def ensure_bind_mount_host_dirs(mounts) -> None:
    """Pre-create missing bind-mount host directories with safe permissions.

    Docker daemon auto-creates missing bind-mount source directories as
    root:root (uid=0, gid=0, mode=0o755).  Any container running as a
    non-root user will then get Permission Denied when it tries to write
    into that path.

    This function creates missing directories *before* ``docker run`` is
    called, using the current process's user (the TRION service user) with
    mode 0o750.  Directories that already exist are left completely
    untouched.
    """
    for mount in (mounts or []):
        mount_type = str(getattr(mount, "type", "bind") or "bind").strip().lower()
        if mount_type != "bind":
            continue
        host_raw = str(getattr(mount, "host", "") or "").strip()
        if not host_raw:
            continue
        host_abs = os.path.abspath(host_raw)
        if not os.path.exists(host_abs):
            try:
                os.makedirs(host_abs, mode=0o750, exist_ok=True)
                logger.info(
                    f"[MountUtils] Pre-created bind-mount dir: {host_abs} (mode=0o750) "
                    "to prevent Docker root:root auto-creation"
                )
            except Exception as e:
                logger.warning(
                    f"[MountUtils] Could not pre-create bind-mount dir '{host_abs}': {e}. "
                    "Docker may create it as root:root — container may get Permission Denied."
                )
