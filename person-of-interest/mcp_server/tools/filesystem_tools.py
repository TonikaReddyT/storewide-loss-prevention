"""Filesystem MCP tools.

Provides sandboxed file system operations constrained to the configured
MCP_FILESYSTEM_ROOT directory. All paths are resolved and validated before
any operation to prevent directory traversal and symlink escapes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.filesystem")

# Files/dirs that must never be readable or writable
_BLOCKED_NAMES = {".env", ".env.bak", "secrets", ".git"}


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register filesystem tools on the MCP server."""

    root = Path(cfg.filesystem_root).resolve()

    def _safe_path(relative_or_abs: str) -> Path:
        """Resolve path and assert it is inside the sandbox root.

        Raises:
            PermissionError: If the resolved path escapes the sandbox.
            ValueError: If the path targets a blocked name.
        """
        p = Path(relative_or_abs)
        if not p.is_absolute():
            p = root / p
        resolved = p.resolve()
        # Symlink-safe containment check
        try:
            resolved.relative_to(root)
        except ValueError:
            raise PermissionError(
                f"Path '{relative_or_abs}' is outside the sandbox root '{root}'"
            )
        # Block sensitive names anywhere in the path components
        for part in resolved.parts:
            if part in _BLOCKED_NAMES:
                raise PermissionError(f"Access to '{part}' is not allowed")
        return resolved

    @mcp.tool()
    def fs_list_directory(path: str = ".") -> dict:
        """List the contents of a directory within the sandbox.

        Args:
            path: Relative or absolute path inside the sandbox root.
                  Defaults to the sandbox root itself.

        Returns:
            Dict with 'path' (resolved), 'entries' list of dicts containing
            name, type ('file'|'dir'), and size (for files).
        """
        try:
            p = _safe_path(path)
            if not p.exists():
                return {"error": f"Path does not exist: {path}"}
            if not p.is_dir():
                return {"error": f"Path is not a directory: {path}"}
            entries = []
            for item in sorted(p.iterdir()):
                entries.append(
                    {
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    }
                )
            return {"path": str(p), "entries": entries}
        except (PermissionError, ValueError) as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def fs_read_file(path: str, encoding: str = "utf-8", max_bytes: int = 1_048_576) -> dict:
        """Read the contents of a file within the sandbox.

        Args:
            path: Relative or absolute path inside the sandbox root.
            encoding: Text encoding (default 'utf-8'). Use 'binary' to get
                      a hex representation of binary files.
            max_bytes: Maximum bytes to read (default 1 MB).

        Returns:
            Dict with 'path', 'content' (string), and 'size' (bytes).
        """
        try:
            p = _safe_path(path)
            if not p.exists():
                return {"error": f"File does not exist: {path}"}
            if not p.is_file():
                return {"error": f"Path is not a file: {path}"}
            size = p.stat().st_size
            if encoding == "binary":
                data = p.read_bytes()[:max_bytes]
                return {"path": str(p), "content": data.hex(), "size": size, "encoding": "hex"}
            content = p.read_text(encoding=encoding, errors="replace")
            if len(content.encode(encoding, errors="replace")) > max_bytes:
                content = content[: max_bytes // 4] + "\n... [TRUNCATED]"
            return {"path": str(p), "content": content, "size": size}
        except (PermissionError, ValueError) as exc:
            return {"error": str(exc)}
        except OSError as exc:
            return {"error": f"OS error reading file: {exc}"}

    @mcp.tool()
    def fs_write_file(path: str, content: str, encoding: str = "utf-8") -> dict:
        """Write content to a file within the sandbox.

        Creates parent directories as needed. Overwrites existing files.
        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            path: Relative or absolute path inside the sandbox root.
            content: Text content to write.
            encoding: File encoding (default 'utf-8').

        Returns:
            Dict with 'path', 'bytes_written', and 'status'.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        try:
            p = _safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding=encoding)
            return {"path": str(p), "bytes_written": len(content.encode(encoding)), "status": "written"}
        except (PermissionError, ValueError) as exc:
            return {"error": str(exc)}
        except OSError as exc:
            return {"error": f"OS error writing file: {exc}"}

    @mcp.tool()
    def fs_get_file_info(path: str) -> dict:
        """Get metadata for a file or directory within the sandbox.

        Args:
            path: Relative or absolute path inside the sandbox root.

        Returns:
            Dict with path, type, size, modified_at, created_at, permissions.
        """
        try:
            p = _safe_path(path)
            if not p.exists():
                return {"error": f"Path does not exist: {path}"}
            stat = p.stat()
            return {
                "path": str(p),
                "type": "dir" if p.is_dir() else "file",
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
                "created_at": stat.st_ctime,
                "permissions": oct(stat.st_mode)[-3:],
                "is_symlink": p.is_symlink(),
            }
        except (PermissionError, ValueError) as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def fs_search_files(pattern: str, base_path: str = ".") -> list[dict]:
        """Search for files matching a glob pattern within the sandbox.

        Args:
            pattern: Glob pattern, e.g. '**/*.py' or 'backend/**/*.json'.
            base_path: Directory to start searching from (default: sandbox root).

        Returns:
            List of dicts with path and size for each matching file.
        """
        try:
            base = _safe_path(base_path)
            if not base.is_dir():
                return [{"error": f"Base path is not a directory: {base_path}"}]
            results = []
            for p in base.glob(pattern):
                try:
                    safe = _safe_path(str(p))
                    if safe.is_file():
                        results.append({"path": str(safe), "size": safe.stat().st_size})
                except (PermissionError, ValueError):
                    continue
                if len(results) >= 200:
                    results.append({"note": "Results truncated at 200 entries"})
                    break
            return results
        except (PermissionError, ValueError) as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def fs_delete_file(path: str) -> dict:
        """Delete a file within the sandbox (directories are not deleted).

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            path: Relative or absolute path to the file inside the sandbox root.

        Returns:
            Confirmation dict or error.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        try:
            p = _safe_path(path)
            if not p.exists():
                return {"error": f"File does not exist: {path}"}
            if not p.is_file():
                return {"error": "Only individual files can be deleted, not directories"}
            p.unlink()
            return {"status": "deleted", "path": str(p)}
        except (PermissionError, ValueError) as exc:
            return {"error": str(exc)}
        except OSError as exc:
            return {"error": f"OS error deleting file: {exc}"}

    log.info("Filesystem tools registered (root=%s, mutations=%s)", root, cfg.allow_mutations)
