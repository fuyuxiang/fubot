"""Echo Agent configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ── Channel configs ──────────────────────────────────────────────────────────

class TelegramChannelConfig(_Base):
    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    proxy: str | None = None
    group_policy: Literal["open", "mention"] = "mention"
    reactions_enabled: bool = True


class DiscordChannelConfig(_Base):
    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    group_policy: Literal["open", "mention"] = "mention"
    reactions_enabled: bool = True


class WebhookChannelConfig(_Base):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    secret: str = ""
    path: str = "/webhook"


class CLIChannelConfig(_Base):
    enabled: bool = True


class CronChannelConfig(_Base):
    enabled: bool = False


class SlackChannelConfig(_Base):
    enabled: bool = False
    bot_token: str = ""
    app_token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    reactions_enabled: bool = True


class WhatsAppChannelConfig(_Base):
    enabled: bool = False
    verify_token: str = ""
    access_token: str = ""
    phone_number_id: str = ""
    webhook_path: str = "/whatsapp"
    host: str = "0.0.0.0"
    port: int = 8081


class WeixinChannelConfig(_Base):
    enabled: bool = False
    account_id: str = ""
    token: str = ""
    base_url: str = "https://ilinkai.weixin.qq.com"
    cdn_base_url: str = "https://novac2c.cdn.weixin.qq.com/c2c"
    allow_from: list[str] = Field(default_factory=list)
    dm_policy: str = "open"
    data_dir: str = ""


class QQBotChannelConfig(_Base):
    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    allow_from: list[str] = Field(default_factory=list)
    sandbox: bool = False
    markdown_support: bool = False
    media_enabled: bool = True
    media_max_file_size_mb: int = 20
    media_upload_cache_size: int = 500
    media_parse_tags: bool = True


class FeishuChannelConfig(_Base):
    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    verification_token: str = ""
    encryption_key: str = ""
    webhook_path: str = "/feishu"
    host: str = "0.0.0.0"
    port: int = 8083


class DingTalkChannelConfig(_Base):
    enabled: bool = False
    app_key: str = ""
    app_secret: str = ""
    robot_code: str = ""
    allow_from: list[str] = Field(default_factory=list)


class EmailChannelConfig(_Base):
    enabled: bool = False
    imap_host: str = ""
    imap_port: int = 993
    smtp_host: str = ""
    smtp_port: int = 465
    username: str = ""
    password: str = ""
    use_ssl: bool = True
    poll_interval_seconds: int = 30
    allow_from: list[str] = Field(default_factory=list)


class WeComChannelConfig(_Base):
    enabled: bool = False
    corp_id: str = ""
    agent_id: str = ""
    secret: str = ""
    token: str = ""
    encoding_aes_key: str = ""
    webhook_path: str = "/wecom"
    host: str = "0.0.0.0"
    port: int = 8084


class MatrixChannelConfig(_Base):
    enabled: bool = False
    homeserver: str = ""
    user_id: str = ""
    access_token: str = ""
    allow_rooms: list[str] = Field(default_factory=list)
    reactions_enabled: bool = True


class ChannelsConfig(_Base):
    telegram: TelegramChannelConfig = Field(default_factory=TelegramChannelConfig)
    discord: DiscordChannelConfig = Field(default_factory=DiscordChannelConfig)
    webhook: WebhookChannelConfig = Field(default_factory=WebhookChannelConfig)
    cli: CLIChannelConfig = Field(default_factory=CLIChannelConfig)
    cron: CronChannelConfig = Field(default_factory=CronChannelConfig)
    slack: SlackChannelConfig = Field(default_factory=SlackChannelConfig)
    whatsapp: WhatsAppChannelConfig = Field(default_factory=WhatsAppChannelConfig)
    weixin: WeixinChannelConfig = Field(default_factory=WeixinChannelConfig)
    qqbot: QQBotChannelConfig = Field(default_factory=QQBotChannelConfig)
    feishu: FeishuChannelConfig = Field(default_factory=FeishuChannelConfig)
    dingtalk: DingTalkChannelConfig = Field(default_factory=DingTalkChannelConfig)
    email: EmailChannelConfig = Field(default_factory=EmailChannelConfig)
    wecom: WeComChannelConfig = Field(default_factory=WeComChannelConfig)
    matrix: MatrixChannelConfig = Field(default_factory=MatrixChannelConfig)
    send_progress: bool = False
    send_tool_hints: bool = False
    stream_channels: list[str] = Field(default_factory=lambda: ["telegram", "discord", "slack"])
    stream_flush_chars: int = 180
    stream_flush_interval_ms: int = 1500
    stream_paragraph_mode: bool = True
    transcription_api_key: str = ""


# ── Provider configs ─────────────────────────────────────────────────────────

class ProviderConfig(_Base):
    name: str = ""
    api_key: str = ""
    api_base: str = ""
    models: list[str] = Field(default_factory=list)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    max_retries: int = 3
    timeout_seconds: int = 120
    rate_limit_rpm: int = 0
    credential_pool: list[str] = Field(default_factory=list)


class ModelRouteConfig(_Base):
    model: str = ""
    provider: str = ""
    task_types: list[str] = Field(default_factory=list)
    fallback_models: list[str] = Field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.7
    context_window: int = 65536


class ModelsConfig(_Base):
    default_model: str = ""
    providers: list[ProviderConfig] = Field(default_factory=list)
    routes: list[ModelRouteConfig] = Field(default_factory=list)
    cost_limit_daily_usd: float = 0.0
    fallback_model: str = ""


# ── Tool configs ─────────────────────────────────────────────────────────────

class ExecToolConfig(_Base):
    enabled: bool = True
    timeout_seconds: int = 30
    max_output_chars: int = 16000
    allowed_commands: list[str] = Field(default_factory=list)
    blocked_commands: list[str] = Field(default_factory=list)


class WebToolConfig(_Base):
    enabled: bool = True
    proxy: str | None = None
    timeout_seconds: int = 30
    search_api_key: str = ""
    search_provider: Literal["brave", "tavily", "serpapi", "searxng"] = "brave"
    search_api_base: str = ""


class ImageGenConfig(_Base):
    api_key: str = ""
    api_base: str = ""
    model: str = "dall-e-3"


class TTSConfig(_Base):
    openai_api_key: str = ""
    default_backend: str = "edge"
    default_voice: str = ""


class CodeExecConfig(_Base):
    enabled: bool = True
    timeout_seconds: int = 30
    allowed_languages: list[str] = Field(default_factory=lambda: ["python", "javascript", "bash"])


class MCPServerConfig(_Base):
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    auth: str = ""
    enabled: bool = True
    timeout: int = 120
    connect_timeout: int = 60
    tools_include: list[str] = Field(default_factory=list)
    tools_exclude: list[str] = Field(default_factory=list)
    transport: Literal["auto", "stdio", "http", "streamable-http"] = "auto"


class ToolsConfig(_Base):
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    web: WebToolConfig = Field(default_factory=WebToolConfig)
    restrict_to_workspace: bool = True
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    mcp_security_policy: Literal["warn", "block"] = "block"
    image_gen: ImageGenConfig = Field(default_factory=ImageGenConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    code_exec: CodeExecConfig = Field(default_factory=CodeExecConfig)


# ── Execution environment configs ────────────────────────────────────────────

class ExecutionConfig(_Base):
    default_executor: Literal["local", "sandbox", "container", "remote"] = "local"
    sandbox_root: str = "/tmp/echo-agent-sandbox"
    container_image: str = ""
    remote_host: str = ""
    remote_user: str = "root"
    remote_key_path: str = ""
    remote_strict_host_key: Literal["no", "accept-new", "yes"] = "accept-new"
    remote_connect_timeout: int = 10
    network_policy: Literal["allow", "deny", "restricted"] = "allow"


# ── Permission configs ───────────────────────────────────────────────────────

class ApprovalConfig(_Base):
    require_approval: list[str] = Field(default_factory=list)
    auto_approve: list[str] = Field(default_factory=list)
    auto_deny: list[str] = Field(default_factory=list)
    default_policy: Literal["approve", "deny", "ask"] = "approve"


class PermissionsConfig(_Base):
    admin_users: list[str] = Field(default_factory=list)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)


class CredentialSecurityConfig(_Base):
    encryption_key_env: str = "ECHO_AGENT_CREDENTIAL_KEY"
    require_encryption: bool = False


# ── Session configs ──────────────────────────────────────────────────────────

class SessionConfig(_Base):
    max_history_messages: int = 500
    expiry_hours: int = 72
    archive_after_hours: int = 168
    context_window_tokens: int = 65536
    introduction_enabled: bool = True
    introduction_template: str = ""


# ── Memory configs ───────────────────────────────────────────────────────────

class MemoryConfig(_Base):
    enabled: bool = True
    consolidation_threshold: int = 50
    vector_enabled: bool = False
    vector_dimensions: int = 1536
    max_user_memories: int = 1000
    max_env_memories: int = 500
    memory_nudge_interval: int = 15
    importance_decay_days: float = 30.0
    snapshot_enabled: bool = True
    graph_enabled: bool = False
    hybrid_retrieval: bool = True
    prefetch_enabled: bool = False
    contradiction_detection: bool = False
    adaptive_forgetting: bool = True
    sleep_consolidation: bool = True
    archival_threshold: float = 0.05
    forget_threshold: float = 0.01
    max_working_memory: int = 20
    max_episodes: int = 500
    embedding_model: str = ""
    embedding_batch_size: int = 32
    consolidation_idle_seconds: int = 300


class KnowledgeConfig(_Base):
    enabled: bool = False
    docs_dir: str = "data/knowledge"
    index_path: str = "data/knowledge_index.json"
    auto_index: bool = True
    chunk_size: int = 1200
    chunk_overlap: int = 120
    max_results: int = 5
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".py"]
    )
    require_citations: bool = True


# ── Multi-agent dispatch configs ─────────────────────────────────────────────

class AgentProfileConfig(_Base):
    id: str = ""
    name: str = ""
    description: str = ""
    instructions: str = ""
    capabilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    tools_allow: list[str] = Field(default_factory=list)
    tools_deny: list[str] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    max_iterations: int = 8
    max_tokens: int = 4096
    temperature: float = 0.4
    priority: int = 0


class MultiAgentConfig(_Base):
    enabled: bool = False
    mode: Literal["assist", "auto"] = "auto"
    default_agent: str = "general"
    route_threshold: float = 0.45
    multi_threshold: float = 0.62
    max_parallel_agents: int = 3
    max_iterations: int = 8
    synthesize_results: bool = True
    audit_path: str = "data/multi_agent_dispatch.jsonl"
    agents: list[AgentProfileConfig] = Field(default_factory=list)


# ── Scheduler configs ───────────────────────────────────────────────────────

class SchedulerConfig(_Base):
    enabled: bool = True
    max_concurrent_jobs: int = 10
    dead_task_timeout_seconds: int = 3600


# ── Storage configs ──────────────────────────────────────────────────────────

class StorageConfig(_Base):
    backend: Literal["sqlite", "filesystem"] = "sqlite"
    database_path: str = "data/echo_agent.db"
    sessions_dir: str = "data/sessions"
    memory_dir: str = "data/memory"
    workspace_dir: str = "data/workspace"
    logs_dir: str = "data/logs"


# ── Observability configs ────────────────────────────────────────────────────

class ObservabilityConfig(_Base):
    log_level: str = "INFO"
    trace_enabled: bool = True
    show_tool_calls: bool = True
    show_route_decisions: bool = False
    health_check_interval_seconds: int = 60
    otel_enabled: bool = True
    otel_endpoint: str = ""
    otel_service_name: str = "echo-agent"
    otel_export_interval_ms: int = 5000


# ── Compression configs ──────────────────────────────────────────────────────

class CompressionConfig(_Base):
    enabled: bool = True
    trigger_ratio: float = 0.7
    tail_budget_ratio: float = 0.4
    head_protect_count: int = 3
    summary_target_ratio: float = 0.20
    summary_min_tokens: int = 2000
    summary_max_tokens: int = 12000
    summary_model: str = ""
    summary_cooldown_seconds: int = 600
    tool_pruning_enabled: bool = True
    tool_pruning_tail_budget_ratio: float = 0.3
    max_compression_count: int = 10


# ── Gateway configs ─────────────────────────────────────────────────────────

class GatewaySessionPolicyConfig(_Base):
    mode: Literal["daily", "idle", "both", "none"] = "idle"
    daily_reset_hour: int = 4
    idle_timeout_minutes: int = 1440


class GatewayPlatformConfig(_Base):
    enabled: bool = False
    home_channel: str = ""
    home_chat_id: str = ""
    reply_mode: Literal["off", "first", "all"] = "off"
    rate_limit_rpm: int = 30


class GatewayAuthConfig(_Base):
    mode: Literal["open", "allowlist", "pairing"] = "allowlist"
    allowed_users: list[str] = Field(default_factory=list)
    admin_users: list[str] = Field(default_factory=list)
    api_tokens: list[str] = Field(default_factory=list)
    token_header: str = "X-Echo-Agent-Token"
    pairing_ttl_seconds: int = 300


class GatewayConfig(_Base):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 9000
    api_prefix: str = "/api/v1"
    ws_path: str = "/ws"
    session_policy: GatewaySessionPolicyConfig = Field(default_factory=GatewaySessionPolicyConfig)
    auth: GatewayAuthConfig = Field(default_factory=GatewayAuthConfig)
    platforms: dict[str, GatewayPlatformConfig] = Field(default_factory=dict)
    media_cache_dir: str = "data/media_cache"
    media_cache_max_mb: int = 500
    max_agent_cache_size: int = 50
    enable_progressive_edit: bool = True
    hooks_dir: str = ""


# ── Skills configs ───────────────────────────────────────────────────────────

class SkillsConfig(_Base):
    skills_dir: str = "skills"
    auto_load: list[str] = Field(default_factory=list)
    creation_nudge_interval: int = 10
    disabled: list[str] = Field(default_factory=list)
    platform_disabled: dict[str, list[str]] = Field(default_factory=dict)
    external_dirs: list[str] = Field(default_factory=list)


# ── Root config ──────────────────────────────────────────────────────────────

class PlanningConfig(_Base):
    enabled: bool = True
    default_strategy: str = "auto"
    max_tree_depth: int = 5
    max_branches: int = 3
    reflection_enabled: bool = True


class A2AConfig(_Base):
    enabled: bool = True
    agent_name: str = "echo-agent"
    agent_description: str = "A modular AI agent framework"
    capabilities: list[str] = Field(default_factory=lambda: ["chat", "tool_use"])


class EvalConfig(_Base):
    enabled: bool = True
    dataset_path: str = "data/eval"
    parallel_cases: int = 3
    timeout_per_case: int = 120


class Config(_Base):
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    credentials: CredentialSecurityConfig = Field(default_factory=CredentialSecurityConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    workspace: str = "~/.echo-agent"
