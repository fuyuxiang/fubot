# Security Policy

## Core Rules

- Keep secrets in config files or environment variables, never in source code.
- Use allowlists on every chat channel in production.
- Do not run the service as `root`.
- Prefer `restrictToWorkspace=true` for file and shell tools.
- Keep the local bridge bound to loopback only.

## Runtime Data

- Default local state lives under `~/.fubot`
- The checked-in runnable config for this repository lives at [`runtime/config.json`](/Users/fuyuxiang/Desktop/testt/fubot/runtime/config.json)
- Session logs, workflow logs, media files, and bridge auth state may contain sensitive data

## Tool Safety

- Shell execution has deny patterns for destructive commands.
- Filesystem tools block path traversal and can be workspace-restricted.
- Coordinator and executors use profile-scoped tool allowlists.
- MCP tool calls have timeouts and explicit registration.

## Channel Safety

- Empty `allowFrom` means deny all.
- `["*"]` means explicitly allow all.
- Group mention/open/allowlist semantics are channel-specific and preserved.
- Email requires explicit `consentGranted=true`.

## Provider Safety

- Provider selection is config-driven.
- Routing health cache avoids repeatedly sending traffic to failing providers.
- Retry logic is limited to transient errors.
- OAuth providers are explicit and do not silently become fallback targets.

## Incident Response

1. Revoke affected API keys or OAuth sessions.
2. Inspect workflow logs, session files, and channel logs.
3. Rotate bridge tokens and mailbox credentials.
4. Re-run tests and startup checks before resuming service.
