"""Jira MCP tools.

Provides tools for Jira issue management: create, read, update,
list, transition, and comment on issues.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.jira")


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register Jira tools on the MCP server."""

    def _client():
        try:
            from atlassian import Jira
        except ImportError:
            raise RuntimeError(
                "atlassian-python-api not installed. Run: pip install atlassian-python-api"
            )
        if not cfg.jira_url:
            raise ValueError("JIRA_URL env var is not set")
        if not cfg.jira_api_token:
            raise ValueError("JIRA_API_TOKEN env var is not set")
        return Jira(
            url=cfg.jira_url,
            username=cfg.jira_username,
            password=cfg.jira_api_token,
            cloud=True,
        )

    @mcp.tool()
    def jira_get_issue(issue_key: str) -> dict:
        """Get details of a Jira issue.

        Args:
            issue_key: The Jira issue key, e.g. 'POI-123'.

        Returns:
            Issue details including summary, description, status, assignee,
            priority, labels, and created/updated timestamps.
        """
        jira = _client()
        try:
            issue = jira.issue(issue_key)
            fields = issue.get("fields", {})
            return {
                "key": issue.get("key"),
                "summary": fields.get("summary", ""),
                "description": _extract_text(fields.get("description")),
                "status": fields.get("status", {}).get("name", ""),
                "assignee": (fields.get("assignee") or {}).get("displayName", "Unassigned"),
                "reporter": (fields.get("reporter") or {}).get("displayName", ""),
                "priority": (fields.get("priority") or {}).get("name", ""),
                "labels": fields.get("labels", []),
                "issue_type": fields.get("issuetype", {}).get("name", ""),
                "project": fields.get("project", {}).get("key", ""),
                "created": fields.get("created", ""),
                "updated": fields.get("updated", ""),
                "url": f"{cfg.jira_url}/browse/{issue.get('key')}",
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def jira_list_issues(
        jql: str = "",
        project: str = "",
        status: str = "",
        assignee: str = "",
        limit: int = 20,
    ) -> list[dict]:
        """List Jira issues using JQL or field filters.

        Args:
            jql: JQL query string. Takes precedence over field filters.
            project: Project key filter (uses JIRA_PROJECT_KEY if empty and jql is empty).
            status: Status filter, e.g. 'In Progress', 'To Do'.
            assignee: Assignee username or 'currentUser()'.
            limit: Maximum number of issues to return (default 20, max 50).

        Returns:
            List of issue summaries with key, summary, status, and assignee.
        """
        jira = _client()
        try:
            if not jql:
                parts = []
                proj = project or cfg.jira_project_key
                if proj:
                    parts.append(f"project = {proj}")
                if status:
                    parts.append(f'status = "{status}"')
                if assignee:
                    parts.append(f"assignee = {assignee}")
                parts.append("ORDER BY updated DESC")
                jql = " AND ".join(parts) if len(parts) > 1 else parts[0] if parts else "ORDER BY updated DESC"

            result = jira.jql(jql, limit=min(limit, 50))
            issues = []
            for issue in result.get("issues", []):
                fields = issue.get("fields", {})
                issues.append(
                    {
                        "key": issue.get("key"),
                        "summary": fields.get("summary", ""),
                        "status": fields.get("status", {}).get("name", ""),
                        "assignee": (fields.get("assignee") or {}).get(
                            "displayName", "Unassigned"
                        ),
                        "priority": (fields.get("priority") or {}).get("name", ""),
                        "updated": fields.get("updated", ""),
                        "url": f"{cfg.jira_url}/browse/{issue.get('key')}",
                    }
                )
            return issues
        except Exception as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def jira_create_issue(
        summary: str,
        issue_type: str = "Task",
        description: str = "",
        project: str = "",
        priority: str = "Medium",
        labels: str = "",
        assignee: str = "",
    ) -> dict:
        """Create a new Jira issue.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            summary: Issue summary / title.
            issue_type: Issue type, e.g. 'Task', 'Bug', 'Story'. Default: 'Task'.
            description: Issue description text.
            project: Project key. Uses JIRA_PROJECT_KEY env var if not supplied.
            priority: Priority name, e.g. 'Low', 'Medium', 'High'. Default: 'Medium'.
            labels: Comma-separated label strings.
            assignee: Assignee account ID or username.

        Returns:
            Created issue key and URL.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        jira = _client()
        proj = project or cfg.jira_project_key
        if not proj:
            return {"error": "No project key provided and JIRA_PROJECT_KEY is not set"}
        try:
            fields: dict = {
                "project": {"key": proj},
                "summary": summary,
                "issuetype": {"name": issue_type},
                "priority": {"name": priority},
            }
            if description:
                fields["description"] = {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description}],
                        }
                    ],
                }
            if labels:
                fields["labels"] = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
            if assignee:
                fields["assignee"] = {"name": assignee}

            result = jira.create_issue(fields=fields)
            key = result.get("key", "")
            return {"key": key, "url": f"{cfg.jira_url}/browse/{key}"}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def jira_update_issue(
        issue_key: str,
        summary: str = "",
        description: str = "",
        priority: str = "",
        labels: str = "",
        assignee: str = "",
    ) -> dict:
        """Update fields on an existing Jira issue.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            issue_key: The Jira issue key, e.g. 'POI-123'.
            summary: New summary (leave empty to keep existing).
            description: New description text (leave empty to keep existing).
            priority: New priority, e.g. 'High' (leave empty to keep existing).
            labels: Comma-separated labels to replace existing labels.
            assignee: New assignee username (leave empty to keep existing).

        Returns:
            Success confirmation or error.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        jira = _client()
        try:
            fields: dict = {}
            if summary:
                fields["summary"] = summary
            if description:
                fields["description"] = {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description}],
                        }
                    ],
                }
            if priority:
                fields["priority"] = {"name": priority}
            if labels:
                fields["labels"] = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
            if assignee:
                fields["assignee"] = {"name": assignee}

            if not fields:
                return {"error": "No fields provided to update"}

            jira.update_issue_field(issue_key, fields)
            return {"status": "updated", "key": issue_key}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def jira_transition_issue(issue_key: str, transition_name: str) -> dict:
        """Transition a Jira issue to a new status.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            issue_key: The Jira issue key, e.g. 'POI-123'.
            transition_name: Target status name, e.g. 'In Progress', 'Done'.

        Returns:
            Success confirmation or error with available transitions.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        jira = _client()
        try:
            transitions = jira.get_issue_transitions(issue_key)
            match = next(
                (
                    t
                    for t in transitions
                    if t.get("name", "").lower() == transition_name.lower()
                ),
                None,
            )
            if match is None:
                available = [t.get("name") for t in transitions]
                return {
                    "error": f"Transition '{transition_name}' not found",
                    "available_transitions": available,
                }
            jira.issue_transition(issue_key, match["id"])
            return {"status": "transitioned", "key": issue_key, "new_status": transition_name}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def jira_add_comment(issue_key: str, comment: str) -> dict:
        """Add a comment to a Jira issue.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            issue_key: The Jira issue key, e.g. 'POI-123'.
            comment: Comment text.

        Returns:
            Comment ID and URL.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        jira = _client()
        try:
            result = jira.issue_add_comment(issue_key, comment)
            return {
                "id": result.get("id"),
                "url": f"{cfg.jira_url}/browse/{issue_key}",
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def jira_get_transitions(issue_key: str) -> list[dict]:
        """List available workflow transitions for a Jira issue.

        Args:
            issue_key: The Jira issue key, e.g. 'POI-123'.

        Returns:
            List of available transitions with id and name.
        """
        jira = _client()
        try:
            transitions = jira.get_issue_transitions(issue_key)
            return [{"id": t.get("id"), "name": t.get("name")} for t in transitions]
        except Exception as exc:
            return [{"error": str(exc)}]

    log.info("Jira tools registered (mutations=%s)", cfg.allow_mutations)


def _extract_text(content) -> str:
    """Extract plain text from Jira Atlassian Document Format or string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = []
        for block in content.get("content", []):
            for inline in block.get("content", []):
                if inline.get("type") == "text":
                    parts.append(inline.get("text", ""))
        return " ".join(parts)
    return str(content)
