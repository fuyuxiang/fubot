# Tool Notes

Tool signatures are exposed automatically.

## Important Constraints

- `exec` is guarded and may be workspace-restricted.
- `message` is for proactive delivery, not normal final replies.
- `cron` creates durable scheduled work.
- MCP tools are loaded dynamically and may time out.

## Multi-Agent Reminder

Executors may have narrower tool access than the coordinator. If a tool is missing, choose a different approach or let the coordinator reroute the task.
