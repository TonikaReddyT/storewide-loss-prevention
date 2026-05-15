"""Docker MCP tools.

Provides tools for Docker container management: list, inspect, start, stop,
get logs, execute commands, and retrieve resource statistics.

Read operations are always available. Mutating operations (start, stop, exec)
require MCP_ALLOW_MUTATIONS=true.
"""

from __future__ import annotations

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.docker")


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register Docker tools on the MCP server."""

    def _client():
        try:
            import docker
        except ImportError:
            raise RuntimeError("docker SDK not installed. Run: pip install docker")
        if cfg.docker_base_url:
            return docker.DockerClient(base_url=cfg.docker_base_url)
        return docker.from_env()

    @mcp.tool()
    def docker_list_containers(all_containers: bool = False) -> list[dict]:
        """List Docker containers.

        Args:
            all_containers: If True, include stopped containers. Default: False (running only).

        Returns:
            List of container dicts with id, name, image, status, ports, and created.
        """
        client = _client()
        try:
            containers = client.containers.list(all=all_containers)
            return [
                {
                    "id": c.short_id,
                    "name": c.name,
                    "image": c.image.tags[0] if c.image.tags else c.image.short_id,
                    "status": c.status,
                    "ports": c.ports,
                    "created": str(c.attrs.get("Created", "")),
                    "labels": c.labels,
                }
                for c in containers
            ]
        except Exception as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def docker_get_container(container_id_or_name: str) -> dict:
        """Get detailed information about a Docker container.

        Args:
            container_id_or_name: Container ID (short or full) or name.

        Returns:
            Dict with id, name, image, status, state, ports, mounts,
            environment variables, and network settings.
        """
        client = _client()
        try:
            c = client.containers.get(container_id_or_name)
            attrs = c.attrs
            state = attrs.get("State", {})
            config = attrs.get("Config", {})
            return {
                "id": c.short_id,
                "name": c.name,
                "image": c.image.tags[0] if c.image.tags else c.image.short_id,
                "status": c.status,
                "state": {
                    "running": state.get("Running"),
                    "paused": state.get("Paused"),
                    "restarting": state.get("Restarting"),
                    "exit_code": state.get("ExitCode"),
                    "started_at": state.get("StartedAt"),
                    "finished_at": state.get("FinishedAt"),
                },
                "ports": c.ports,
                "mounts": [
                    {"source": m.get("Source"), "destination": m.get("Destination"), "mode": m.get("Mode")}
                    for m in attrs.get("Mounts", [])
                ],
                "env": config.get("Env", []),
                "restart_policy": attrs.get("HostConfig", {}).get("RestartPolicy", {}),
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def docker_get_logs(
        container_id_or_name: str,
        tail: int = 100,
        since_minutes: int = 0,
    ) -> dict:
        """Get recent log output from a Docker container.

        Args:
            container_id_or_name: Container ID or name.
            tail: Number of log lines from the end (default 100, max 500).
            since_minutes: Return only logs from the last N minutes (0 = no filter).

        Returns:
            Dict with container_name, logs string, and line_count.
        """
        client = _client()
        try:
            c = client.containers.get(container_id_or_name)
            kwargs: dict = {
                "stdout": True,
                "stderr": True,
                "timestamps": True,
                "tail": min(tail, 500),
            }
            if since_minutes > 0:
                import datetime
                kwargs["since"] = datetime.datetime.utcnow() - datetime.timedelta(minutes=since_minutes)
            raw_logs = c.logs(**kwargs)
            if isinstance(raw_logs, bytes):
                log_str = raw_logs.decode("utf-8", errors="replace")
            else:
                log_str = str(raw_logs)
            lines = log_str.splitlines()
            return {
                "container_name": c.name,
                "logs": log_str[-32768:] if len(log_str) > 32768 else log_str,
                "line_count": len(lines),
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def docker_get_stats(container_id_or_name: str) -> dict:
        """Get real-time resource usage statistics for a Docker container.

        Args:
            container_id_or_name: Container ID or name.

        Returns:
            Dict with cpu_percent, memory_usage_mb, memory_limit_mb,
            memory_percent, net_io, and block_io.
        """
        client = _client()
        try:
            c = client.containers.get(container_id_or_name)
            stats = c.stats(stream=False)

            # CPU percentage
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            sys_delta = (
                stats["cpu_stats"].get("system_cpu_usage", 0)
                - stats["precpu_stats"].get("system_cpu_usage", 0)
            )
            num_cpus = stats["cpu_stats"].get("online_cpus") or len(
                stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1])
            )
            cpu_pct = (cpu_delta / sys_delta) * num_cpus * 100.0 if sys_delta > 0 else 0.0

            # Memory
            mem = stats.get("memory_stats", {})
            mem_usage_mb = mem.get("usage", 0) / 1_048_576
            mem_limit_mb = mem.get("limit", 0) / 1_048_576
            mem_pct = (mem_usage_mb / mem_limit_mb * 100) if mem_limit_mb > 0 else 0.0

            # Network I/O
            networks = stats.get("networks", {})
            net_rx = sum(v.get("rx_bytes", 0) for v in networks.values())
            net_tx = sum(v.get("tx_bytes", 0) for v in networks.values())

            # Block I/O
            blk_io = stats.get("blkio_stats", {}).get("io_service_bytes_recursive") or []
            blk_read = sum(b.get("value", 0) for b in blk_io if b.get("op") == "Read")
            blk_write = sum(b.get("value", 0) for b in blk_io if b.get("op") == "Write")

            return {
                "container_name": c.name,
                "cpu_percent": round(cpu_pct, 2),
                "memory_usage_mb": round(mem_usage_mb, 2),
                "memory_limit_mb": round(mem_limit_mb, 2),
                "memory_percent": round(mem_pct, 2),
                "net_io": {
                    "rx_mb": round(net_rx / 1_048_576, 3),
                    "tx_mb": round(net_tx / 1_048_576, 3),
                },
                "block_io": {
                    "read_mb": round(blk_read / 1_048_576, 3),
                    "write_mb": round(blk_write / 1_048_576, 3),
                },
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def docker_start_container(container_id_or_name: str) -> dict:
        """Start a stopped Docker container.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            container_id_or_name: Container ID or name.

        Returns:
            Confirmation dict with container name and new status.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        client = _client()
        try:
            c = client.containers.get(container_id_or_name)
            c.start()
            c.reload()
            return {"status": c.status, "name": c.name, "action": "started"}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def docker_stop_container(container_id_or_name: str, timeout: int = 10) -> dict:
        """Stop a running Docker container.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            container_id_or_name: Container ID or name.
            timeout: Seconds to wait before forcibly killing. Default: 10.

        Returns:
            Confirmation dict with container name and new status.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        client = _client()
        try:
            c = client.containers.get(container_id_or_name)
            c.stop(timeout=timeout)
            c.reload()
            return {"status": c.status, "name": c.name, "action": "stopped"}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def docker_exec_command(
        container_id_or_name: str,
        command: str,
        workdir: str = "",
        user: str = "",
    ) -> dict:
        """Execute a command inside a running Docker container.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            container_id_or_name: Container ID or name.
            command: Shell command to execute (runs via /bin/sh -c).
            workdir: Working directory inside the container (optional).
            user: User to run the command as (optional).

        Returns:
            Dict with exit_code and output string.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        client = _client()
        try:
            c = client.containers.get(container_id_or_name)
            kwargs: dict = {"cmd": ["/bin/sh", "-c", command]}
            if workdir:
                kwargs["workdir"] = workdir
            if user:
                kwargs["user"] = user
            exit_code, output = c.exec_run(**kwargs)
            output_str = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
            return {
                "container": c.name,
                "command": command,
                "exit_code": exit_code,
                "output": output_str[-8192:] if len(output_str) > 8192 else output_str,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def docker_list_images() -> list[dict]:
        """List Docker images available on the host.

        Returns:
            List of image dicts with id, tags, size_mb, and created.
        """
        client = _client()
        try:
            images = client.images.list()
            return [
                {
                    "id": img.short_id,
                    "tags": img.tags,
                    "size_mb": round(img.attrs.get("Size", 0) / 1_048_576, 1),
                    "created": img.attrs.get("Created", ""),
                }
                for img in images
            ]
        except Exception as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def docker_get_system_info() -> dict:
        """Get Docker system information (version, resources, container counts).

        Returns:
            Dict with docker_version, containers, running, paused, stopped,
            images, memory_total_gb, and cpus.
        """
        client = _client()
        try:
            info = client.info()
            return {
                "docker_version": client.version().get("Version"),
                "containers": info.get("Containers"),
                "running": info.get("ContainersRunning"),
                "paused": info.get("ContainersPaused"),
                "stopped": info.get("ContainersStopped"),
                "images": info.get("Images"),
                "memory_total_gb": round(info.get("MemTotal", 0) / 1_073_741_824, 2),
                "cpus": info.get("NCPU"),
                "os": info.get("OperatingSystem"),
                "kernel": info.get("KernelVersion"),
            }
        except Exception as exc:
            return {"error": str(exc)}

    log.info("Docker tools registered (mutations=%s)", cfg.allow_mutations)
