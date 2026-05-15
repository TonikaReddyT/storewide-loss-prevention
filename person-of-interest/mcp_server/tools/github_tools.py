"""GitHub MCP tools.

Provides tools for GitHub repository, issue, pull request,
file content, and code search operations.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.github")


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register GitHub tools on the MCP server."""

    def _client():
        try:
            from github import Github, GithubException
        except ImportError:
            raise RuntimeError("PyGithub not installed. Run: pip install PyGithub")
        if not cfg.github_token:
            raise ValueError("GITHUB_TOKEN env var is not set")
        return Github(cfg.github_token), GithubException

    @mcp.tool()
    def github_list_repos(org: str = "", user: str = "") -> list[dict]:
        """List GitHub repositories for an organization or user.

        Args:
            org: GitHub organization name. Defaults to GITHUB_ORG env var.
            user: GitHub username. Used when org is not supplied.

        Returns:
            List of dicts with name, description, url, stars, language,
            private flag, and default_branch.
        """
        g, GithubException = _client()
        target = org or cfg.github_org
        try:
            if target:
                entity = g.get_organization(target)
            elif user:
                entity = g.get_user(user)
            else:
                entity = g.get_user()
            return [
                {
                    "name": r.full_name,
                    "description": r.description or "",
                    "url": r.html_url,
                    "stars": r.stargazers_count,
                    "language": r.language or "",
                    "private": r.private,
                    "default_branch": r.default_branch,
                }
                for r in entity.get_repos()
            ]
        except GithubException as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def github_get_issue(repo: str, issue_number: int) -> dict:
        """Get details of a specific GitHub issue.

        Args:
            repo: Full repository name, e.g. 'owner/repo'.
            issue_number: The issue number.

        Returns:
            Issue details including title, body, state, labels, assignees, and URL.
        """
        g, GithubException = _client()
        try:
            issue = g.get_repo(repo).get_issue(issue_number)
            return {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body or "",
                "state": issue.state,
                "labels": [lbl.name for lbl in issue.labels],
                "assignees": [a.login for a in issue.assignees],
                "created_at": issue.created_at.isoformat(),
                "updated_at": issue.updated_at.isoformat(),
                "url": issue.html_url,
                "comments": issue.comments,
            }
        except GithubException as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def github_list_issues(
        repo: str,
        state: str = "open",
        labels: str = "",
        limit: int = 20,
    ) -> list[dict]:
        """List issues for a GitHub repository (excludes pull requests).

        Args:
            repo: Full repository name, e.g. 'owner/repo'.
            state: Filter by state — 'open', 'closed', or 'all'.
            labels: Comma-separated label names to filter by.
            limit: Maximum number of issues to return (default 20, max 100).

        Returns:
            List of issue summaries with number, title, state, labels, and URL.
        """
        g, GithubException = _client()
        try:
            r = g.get_repo(repo)
            label_list = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
            kwargs: dict[str, Any] = {"state": state}
            if label_list:
                kwargs["labels"] = [r.get_label(lbl) for lbl in label_list]
            issues = []
            for issue in r.get_issues(**kwargs):
                if issue.pull_request:
                    continue
                issues.append(
                    {
                        "number": issue.number,
                        "title": issue.title,
                        "state": issue.state,
                        "labels": [lbl.name for lbl in issue.labels],
                        "url": issue.html_url,
                        "created_at": issue.created_at.isoformat(),
                    }
                )
                if len(issues) >= min(limit, 100):
                    break
            return issues
        except GithubException as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def github_create_issue(
        repo: str,
        title: str,
        body: str = "",
        labels: str = "",
        assignees: str = "",
    ) -> dict:
        """Create a GitHub issue.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            repo: Full repository name, e.g. 'owner/repo'.
            title: Issue title.
            body: Issue body (supports Markdown).
            labels: Comma-separated label names to apply.
            assignees: Comma-separated GitHub usernames to assign.

        Returns:
            Created issue with number, title, url, and state.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        g, GithubException = _client()
        try:
            r = g.get_repo(repo)
            issue = r.create_issue(
                title=title,
                body=body,
                labels=[lbl.strip() for lbl in labels.split(",") if lbl.strip()],
                assignees=[a.strip() for a in assignees.split(",") if a.strip()],
            )
            return {
                "number": issue.number,
                "title": issue.title,
                "url": issue.html_url,
                "state": issue.state,
            }
        except GithubException as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def github_add_issue_comment(repo: str, issue_number: int, comment: str) -> dict:
        """Add a comment to a GitHub issue or pull request.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            repo: Full repository name, e.g. 'owner/repo'.
            issue_number: The issue or PR number.
            comment: Comment body (Markdown supported).

        Returns:
            Comment details with id and url.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        g, GithubException = _client()
        try:
            issue = g.get_repo(repo).get_issue(issue_number)
            c = issue.create_comment(comment)
            return {"id": c.id, "url": c.html_url}
        except GithubException as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def github_list_prs(repo: str, state: str = "open", limit: int = 20) -> list[dict]:
        """List pull requests for a GitHub repository.

        Args:
            repo: Full repository name, e.g. 'owner/repo'.
            state: Filter by state — 'open', 'closed', or 'all'.
            limit: Maximum number of PRs to return (default 20, max 100).

        Returns:
            List of PR summaries with number, title, author, branches, and URL.
        """
        g, GithubException = _client()
        try:
            prs = []
            for pr in g.get_repo(repo).get_pulls(
                state=state, sort="updated", direction="desc"
            ):
                prs.append(
                    {
                        "number": pr.number,
                        "title": pr.title,
                        "state": pr.state,
                        "author": pr.user.login,
                        "base": pr.base.ref,
                        "head": pr.head.ref,
                        "url": pr.html_url,
                        "created_at": pr.created_at.isoformat(),
                        "draft": pr.draft,
                    }
                )
                if len(prs) >= min(limit, 100):
                    break
            return prs
        except GithubException as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def github_get_file(repo: str, path: str, ref: str = "") -> dict:
        """Get the decoded contents of a file from a GitHub repository.

        Args:
            repo: Full repository name, e.g. 'owner/repo'.
            path: Path to the file within the repository.
            ref: Branch name, tag, or commit SHA. Defaults to the default branch.

        Returns:
            Dict with content (decoded UTF-8 string), sha, and size.
        """
        g, GithubException = _client()
        try:
            r = g.get_repo(repo)
            kwargs: dict[str, Any] = {"path": path}
            if ref:
                kwargs["ref"] = ref
            item = r.get_contents(**kwargs)
            if isinstance(item, list):
                return {"error": f"'{path}' is a directory. Specify a file path."}
            return {
                "path": item.path,
                "content": item.decoded_content.decode("utf-8", errors="replace"),
                "sha": item.sha,
                "size": item.size,
            }
        except GithubException as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def github_search_code(query: str, repo: str = "", limit: int = 10) -> list[dict]:
        """Search code on GitHub using GitHub's code search syntax.

        Args:
            query: Search query (GitHub code search syntax, e.g. 'faiss repo:owner/repo').
            repo: Optional repository to restrict the search, e.g. 'owner/repo'.
            limit: Maximum number of results (default 10, max 30).

        Returns:
            List of code results with path, repository, and URL.
        """
        g, GithubException = _client()
        try:
            q = f"{query} repo:{repo}" if repo else query
            results = []
            for item in g.search_code(q):
                results.append(
                    {
                        "path": item.path,
                        "repo": item.repository.full_name,
                        "url": item.html_url,
                        "sha": item.sha,
                    }
                )
                if len(results) >= min(limit, 30):
                    break
            return results
        except GithubException as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def github_list_commits(repo: str, branch: str = "", path: str = "", limit: int = 20) -> list[dict]:
        """List recent commits for a GitHub repository or file path.

        Args:
            repo: Full repository name, e.g. 'owner/repo'.
            branch: Branch name. Defaults to the default branch.
            path: Optional file/directory path to filter commits.
            limit: Maximum number of commits to return (default 20).

        Returns:
            List of commits with sha, message, author, and date.
        """
        g, GithubException = _client()
        try:
            r = g.get_repo(repo)
            kwargs: dict[str, Any] = {}
            if branch:
                kwargs["sha"] = branch
            if path:
                kwargs["path"] = path
            commits = []
            for c in r.get_commits(**kwargs):
                commits.append(
                    {
                        "sha": c.sha[:8],
                        "message": (c.commit.message or "").split("\n")[0],
                        "author": c.commit.author.name if c.commit.author else "",
                        "date": c.commit.author.date.isoformat() if c.commit.author else "",
                        "url": c.html_url,
                    }
                )
                if len(commits) >= limit:
                    break
            return commits
        except GithubException as exc:
            return [{"error": str(exc)}]

    log.info("GitHub tools registered (mutations=%s)", cfg.allow_mutations)
