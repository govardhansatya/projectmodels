"""
GitHub importer service
"""
import aiohttp
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def import_github_profile(username: str) -> Dict[str, Any]:
	"""Fetch public GitHub profile and repositories for a username."""
	base = "https://api.github.com"
	async with aiohttp.ClientSession() as session:
		try:
			async with session.get(f"{base}/users/{username}") as resp:
				profile = await resp.json()
			async with session.get(f"{base}/users/{username}/repos?per_page=100&sort=updated") as resp:
				repos = await resp.json()
			# Project summaries
			projects = [
				{
					"name": r.get("name"),
					"description": r.get("description"),
					"technologies": [],
					"github_url": r.get("html_url"),
					"stars": r.get("stargazers_count"),
					"forks": r.get("forks_count"),
					"updated_at": r.get("updated_at"),
				}
				for r in repos if isinstance(repos, list)
			]
			return {"profile": profile, "repos": projects}
		except Exception as e:
			logger.error(f"Failed to import GitHub profile: {e}")
			return {"profile": {}, "repos": []}
