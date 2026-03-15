# Communication

Use this repository as the source of truth for implementation, tests, and runtime behavior.

## Recommended Workflow

1. Open an issue or a task note with the user-facing behavior you want to change.
2. Update or add tests first when the change affects externally visible behavior.
3. Keep architecture notes in [`docs/architecture.md`](/Users/fuyuxiang/Desktop/testt/fubot/docs/architecture.md).
4. Keep feature coverage notes in [`docs/feature-matrix.md`](/Users/fuyuxiang/Desktop/testt/fubot/docs/feature-matrix.md).

## Change Review Checklist

- Does the change preserve existing channel security semantics?
- Does it keep append-only session history intact?
- Does it preserve cron, heartbeat, bridge, and MCP integration paths?
- Does it respect coordinator/executor tool boundaries?
- Does it avoid hardcoding runtime credentials outside the checked-in runtime config?

## Test Expectations

- Run focused regression tests for touched areas.
- Run multi-agent routing tests when orchestration or config changes.
- Run the full suite before release.
