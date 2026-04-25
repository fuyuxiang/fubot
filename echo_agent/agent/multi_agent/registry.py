"""Agent registry for configured specialist agents."""

from __future__ import annotations

from collections.abc import Iterable

from echo_agent.agent.multi_agent.models import AgentProfile
from echo_agent.config.schema import AgentProfileConfig, MultiAgentConfig


class AgentRegistry:
    """Stores agent profiles and exposes capability/tool lookups."""

    def __init__(self, profiles: Iterable[AgentProfile]):
        self._profiles = {profile.id: profile for profile in profiles if profile.id}
        if "general" not in self._profiles:
            self._profiles["general"] = AgentProfile(
                id="general",
                name="General Assistant",
                description="General fallback agent.",
                capabilities=("general", "chat"),
                task_types=("chat",),
            )

    @classmethod
    def from_config(cls, config: MultiAgentConfig) -> "AgentRegistry":
        return cls(_profile_from_config(item) for item in config.agents)

    def get(self, agent_id: str) -> AgentProfile | None:
        return self._profiles.get(agent_id)

    def require(self, agent_id: str) -> AgentProfile:
        profile = self.get(agent_id)
        if not profile:
            raise KeyError(f"Agent profile not found: {agent_id}")
        return profile

    def list(self) -> list[AgentProfile]:
        return sorted(self._profiles.values(), key=lambda p: (-p.priority, p.id))

    def tool_allowed(self, agent_id: str, tool_name: str) -> bool:
        profile = self.require(agent_id)
        if tool_name in profile.tools_deny:
            return False
        if profile.tools_allow and tool_name not in profile.tools_allow:
            return False
        return True

    def filter_tool_names(self, agent_id: str, available: Iterable[str]) -> list[str]:
        return [name for name in available if self.tool_allowed(agent_id, name)]


def _profile_from_config(cfg: AgentProfileConfig) -> AgentProfile:
    return AgentProfile(
        id=cfg.id,
        name=cfg.name or cfg.id,
        description=cfg.description,
        instructions=cfg.instructions,
        capabilities=tuple(cfg.capabilities),
        keywords=tuple(cfg.keywords),
        task_types=tuple(cfg.task_types),
        tools_allow=tuple(cfg.tools_allow),
        tools_deny=tuple(cfg.tools_deny),
        model=cfg.model,
        provider=cfg.provider,
        max_iterations=cfg.max_iterations,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        priority=cfg.priority,
    )
