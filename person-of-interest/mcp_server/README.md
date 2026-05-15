# POI MCP Server

Model Context Protocol (MCP) server for the **Person of Interest (POI) Re-identification System**.

Exposes **9 tool categories** covering the full POI system stack — from GitHub project management to live MQTT event inspection and OpenVINO inference.

---

## Architecture

```
mcp_server/
├── server.py               # FastMCP entry point, lifespan management
├── config.py               # MCPConfig — all settings from env vars
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image
├── .env.example            # Environment variable template
├── claude_desktop_config.json  # Claude Desktop integration config
└── tools/
    ├── github_tools.py     # GitHub API (repos, issues, PRs, files)
    ├── jira_tools.py       # Jira issue management
    ├── filesystem_tools.py # Sandboxed file system access
    ├── redis_tools.py      # POI Redis store inspection
    ├── openvino_tools.py   # OpenVINO face embedding inference
    ├── python_tools.py     # Sandboxed Python code execution
    ├── mqtt_tools.py       # MQTT event buffer + publish
    ├── faiss_tools.py      # FAISS/POI search via backend REST API
    └── docker_tools.py     # Docker container management
```

---

## Tool Categories

### GitHub (`github_*`)
| Tool | Description |
|---|---|
| `github_list_repos` | List repos for an org or user |
| `github_get_issue` | Get issue details |
| `github_list_issues` | List issues with state/label filters |
| `github_create_issue` | Create a new issue *(mutation)* |
| `github_add_issue_comment` | Add comment to issue/PR *(mutation)* |
| `github_list_prs` | List pull requests |
| `github_get_file` | Get decoded file contents from a repo |
| `github_search_code` | Search code across GitHub |
| `github_list_commits` | List recent commits for a branch/path |

### Jira (`jira_*`)
| Tool | Description |
|---|---|
| `jira_get_issue` | Get issue details |
| `jira_list_issues` | Search issues via JQL or field filters |
| `jira_create_issue` | Create a new issue *(mutation)* |
| `jira_update_issue` | Update issue fields *(mutation)* |
| `jira_transition_issue` | Move issue to a new status *(mutation)* |
| `jira_add_comment` | Add a comment *(mutation)* |
| `jira_get_transitions` | List available workflow transitions |

### Filesystem (`fs_*`)
| Tool | Description |
|---|---|
| `fs_list_directory` | List directory contents |
| `fs_read_file` | Read file contents |
| `fs_write_file` | Write file *(mutation)* |
| `fs_get_file_info` | File metadata (size, mtime, permissions) |
| `fs_search_files` | Glob search within sandbox |
| `fs_delete_file` | Delete a file *(mutation)* |

All paths are resolved and validated against `MCP_FILESYSTEM_ROOT` to prevent directory traversal.

### Redis (`redis_*`)
| Tool | Description |
|---|---|
| `redis_list_pois` | List all enrolled POIs |
| `redis_get_poi` | Get full POI details |
| `redis_get_recent_alerts` | Get recent POI match alerts |
| `redis_get_events_for_poi` | Movement events for a POI |
| `redis_get_key` | Get any Redis key value |
| `redis_set_key` | Set a Redis key *(mutation)* |
| `redis_delete_key` | Delete a Redis key *(mutation)* |
| `redis_search_keys` | Pattern-scan Redis keys |
| `redis_get_stats` | Redis server stats + POI/alert counts |
| `redis_flush_alerts` | Clear all alerts *(mutation)* |

### OpenVINO (`openvino_*`)
| Tool | Description |
|---|---|
| `openvino_list_devices` | List available inference devices (CPU, GPU, NPU) |
| `openvino_list_models` | List `.xml` model files in MODEL_BASE |
| `openvino_get_model_info` | Input/output shapes for a model |
| `openvino_generate_face_embedding` | Generate 256-d face embedding from base64 image |
| `openvino_benchmark_inference` | Benchmark inference latency (min/avg/max ms) |

### Python (`python_*`)
| Tool | Description |
|---|---|
| `python_execute` | Execute Python code snippet *(mutation)* |
| `python_run_script` | Run an existing `.py` file *(mutation)* |
| `python_list_packages` | List installed packages |
| `python_get_version` | Python version and platform info |
| `python_run_pip_install` | Install a package via pip *(mutation)* |

⚠️ **Security**: Python execution uses subprocess with env scrubbing and a configurable timeout. This is for development use only — not a hardened sandbox.

### MQTT Events (`mqtt_*`)
| Tool | Description |
|---|---|
| `mqtt_get_recent_events` | Get buffered SceneScape events (no raw embeddings) |
| `mqtt_get_subscriber_status` | Subscriber connection status |
| `mqtt_publish` | Publish a message to any topic *(mutation)* |
| `mqtt_simulate_scenescape_event` | Publish a test person-detection event *(mutation)* |

The subscriber connects on server startup and buffers up to `MQTT_EVENT_BUFFER_SIZE` events (default 100). Raw re-identification embeddings are stripped from the buffer.

### FAISS / POI Management (`faiss_*`, `poi_*`)
| Tool | Description |
|---|---|
| `faiss_get_stats` | Index vector count and server status |
| `faiss_search_by_image` | Search index by uploading a face image |
| `faiss_list_pois` | List POIs with reference image counts |
| `faiss_get_recent_alerts` | Recent POI alerts via backend API |
| `faiss_list_cameras` | Cameras from SceneScape |
| `poi_get` | Get full POI details |
| `poi_delete` | Delete a POI and its FAISS vectors *(mutation)* |
| `poi_create_from_image` | Enroll new POI from base64 image *(mutation)* |

All FAISS operations go through the backend REST API to avoid cross-process index inconsistency.

### Docker (`docker_*`)
| Tool | Description |
|---|---|
| `docker_list_containers` | List containers (running or all) |
| `docker_get_container` | Detailed container info |
| `docker_get_logs` | Container log output |
| `docker_get_stats` | CPU, memory, net, block I/O stats |
| `docker_start_container` | Start a stopped container *(mutation)* |
| `docker_stop_container` | Stop a running container *(mutation)* |
| `docker_exec_command` | Run command inside container *(mutation)* |
| `docker_list_images` | List Docker images |
| `docker_get_system_info` | Docker version and host resource counts |

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp mcp_server/.env.example mcp_server/.env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | — | GitHub personal access token |
| `JIRA_URL` | — | Jira instance URL |
| `JIRA_API_TOKEN` | — | Jira API token |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `POI_BACKEND_URL` | `http://localhost:8000` | POI backend REST API URL |
| `MQTT_HOST` | `localhost` | MQTT broker hostname |
| `MCP_FILESYSTEM_ROOT` | `/workspace/person-of-interest` | Sandbox root for filesystem tools |
| `MCP_ALLOW_MUTATIONS` | `false` | Enable write/mutating tools |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `streamable-http` |
| `MCP_PORT` | `9000` | HTTP port (for streamable-http transport) |

---

## Running

### Docker Compose (recommended)
```bash
# From person-of-interest/
docker compose up mcp-server
```

### Local (stdio — Claude Desktop)
```bash
cd person-of-interest/
pip install -r mcp_server/requirements.txt
PYTHONPATH=. python -m mcp_server.server
```

### Claude Desktop Integration
Copy `mcp_server/claude_desktop_config.json` content into your Claude Desktop
`claude_desktop_config.json`, updating the paths and tokens.

---

## Security Notes

- **MCP_ALLOW_MUTATIONS=false by default** — all write operations are blocked until explicitly enabled.
- **Filesystem sandbox** — all paths are resolved with `Path.resolve()` and must be under `MCP_FILESYSTEM_ROOT`. Symlinks are followed before validation.
- **Python execution** — uses subprocess (not `exec()`), with scrubbed env and configurable timeout. NOT a hardened security boundary.
- **MQTT event buffer** — raw biometric embeddings are stripped; only object metadata is retained.
- **Docker socket** — mounted read-only (`ro`) in the container. Exec and start/stop require mutations.
- **FAISS** — read-only access via backend REST API. No direct cross-process index writes.
