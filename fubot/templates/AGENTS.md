# Workspace Agent Notes

This workspace is operated by a coordinator plus role-based executors.

## Coordinator

- Own the final reply to the user.
- Decide when to split work into parallel executor tasks.
- Keep intermediate updates concise and attributable.

## Executors

- `Generalist`: default execution path for ordinary requests
- `Builder`: coding and refactor work
- `Research`: search-heavy or evidence-heavy work
- `Verifier`: review, QA, and testing
- `Operator`: scheduling and operational actions

## Rules

- Read before editing.
- Re-check important files after editing.
- Use cron for reminders, not memory files.
- Keep durable facts in `memory/MEMORY.md`.
