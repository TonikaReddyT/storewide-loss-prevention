"""Python MCP tools.

Provides a sandboxed Python code execution environment using subprocess.
Execution is time-limited and runs with a scrubbed environment to reduce
accidental exposure of secrets.

SECURITY WARNING: These tools are intended for development and debugging
only. Subprocess isolation prevents in-process damage but does NOT provide
a strong security boundary. Do not expose this server to untrusted clients.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.python")

# Env vars that should never be passed into sandbox subprocesses
_SCRUB_PREFIXES = (
    "GITHUB_TOKEN",
    "JIRA_API_TOKEN",
    "JIRA_USERNAME",
    "MQTT_CA_CERT",
    "SCENESCAPE_API_TOKEN",
    "REDIS_PASSWORD",
    "DATABASE_PASSWORD",
    "SUPASS",
    "CONTROLLER_AUTH",
    "AWS_",
    "AZURE_",
    "GCP_",
)


def _scrubbed_env() -> dict[str, str]:
    """Return a copy of os.environ with sensitive credentials removed."""
    clean = {}
    for key, val in os.environ.items():
        if any(key.startswith(prefix) for prefix in _SCRUB_PREFIXES):
            continue
        clean[key] = val
    clean["PYTHONDONTWRITEBYTECODE"] = "1"
    return clean


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register Python execution tools on the MCP server."""

    timeout = cfg.python_exec_timeout

    @mcp.tool()
    def python_execute(code: str, timeout_secs: int = 0) -> dict:
        """Execute a Python code snippet in a sandboxed subprocess.

        The snippet runs with:
        - A dedicated temporary working directory (deleted after execution)
        - A scrubbed environment (no API tokens or passwords)
        - A configurable timeout (default from MCP_PYTHON_EXEC_TIMEOUT env var)

        SECURITY WARNING: This tool executes arbitrary code. Only expose this
        server to trusted clients. Use MCP_ALLOW_MUTATIONS=true to enable.

        Args:
            code: Python source code to execute.
            timeout_secs: Execution timeout in seconds. 0 = use server default.

        Returns:
            Dict with stdout, stderr, returncode, and timed_out flag.
        """
        if not cfg.allow_mutations:
            return {"error": "Python execution is disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}

        effective_timeout = timeout_secs if timeout_secs > 0 else timeout

        with tempfile.TemporaryDirectory(prefix="poi_mcp_py_") as tmpdir:
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(code, encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                    cwd=tmpdir,
                    env=_scrubbed_env(),
                )
                return {
                    "stdout": result.stdout[-8192:] if len(result.stdout) > 8192 else result.stdout,
                    "stderr": result.stderr[-4096:] if len(result.stderr) > 4096 else result.stderr,
                    "returncode": result.returncode,
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired:
                return {
                    "stdout": "",
                    "stderr": f"Execution timed out after {effective_timeout}s",
                    "returncode": -1,
                    "timed_out": True,
                }

    @mcp.tool()
    def python_run_script(
        script_path: str,
        args: str = "",
        timeout_secs: int = 0,
    ) -> dict:
        """Run an existing Python script file in a sandboxed subprocess.

        Requires MCP_ALLOW_MUTATIONS=true. The script runs from its own
        directory with a scrubbed environment.

        Args:
            script_path: Absolute path to the Python script to run.
            args: Space-separated command-line arguments to pass to the script.
            timeout_secs: Execution timeout in seconds. 0 = use server default.

        Returns:
            Dict with stdout, stderr, returncode, and timed_out flag.
        """
        if not cfg.allow_mutations:
            return {"error": "Script execution is disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}

        p = Path(script_path)
        if not p.exists():
            return {"error": f"Script not found: {script_path}"}
        if not p.is_file():
            return {"error": f"Path is not a file: {script_path}"}

        effective_timeout = timeout_secs if timeout_secs > 0 else timeout
        cmd = [sys.executable, str(p)] + (args.split() if args else [])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=str(p.parent),
                env=_scrubbed_env(),
            )
            return {
                "stdout": result.stdout[-8192:] if len(result.stdout) > 8192 else result.stdout,
                "stderr": result.stderr[-4096:] if len(result.stderr) > 4096 else result.stderr,
                "returncode": result.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {effective_timeout}s",
                "returncode": -1,
                "timed_out": True,
            }

    @mcp.tool()
    def python_list_packages() -> list[dict]:
        """List installed Python packages in the current environment.

        Returns:
            List of dicts with name and version for each installed package.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
                env=_scrubbed_env(),
            )
            if result.returncode != 0:
                return [{"error": result.stderr}]
            import json
            return json.loads(result.stdout)
        except Exception as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def python_get_version() -> dict:
        """Get Python interpreter version and platform information.

        Returns:
            Dict with python_version, platform, executable, and prefix.
        """
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "executable": sys.executable,
            "prefix": sys.prefix,
        }

    @mcp.tool()
    def python_run_pip_install(package: str) -> dict:
        """Install a Python package using pip.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            package: Package name and optional version spec, e.g. 'numpy==1.26.0'.

        Returns:
            Dict with stdout, stderr, and returncode.
        """
        if not cfg.allow_mutations:
            return {"error": "Package installation is disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        # Basic safety check — reject shell metacharacters
        forbidden = set(";&|`$><()\\")
        if any(c in package for c in forbidden):
            return {"error": f"Invalid package spec: contains forbidden characters"}
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=120,
                env=_scrubbed_env(),
            )
            return {
                "stdout": result.stdout[-4096:],
                "stderr": result.stderr[-2048:],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": "pip install timed out after 120s", "returncode": -1}
        except Exception as exc:
            return {"error": str(exc)}

    log.info("Python tools registered (timeout=%ds, mutations=%s)", timeout, cfg.allow_mutations)
