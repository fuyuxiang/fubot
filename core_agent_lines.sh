#!/bin/bash
# Count core agent lines (excluding channels/, cli/, providers/ adapters)
cd "$(dirname "$0")" || exit 1

echo "fubot core agent line count"
echo "================================"
echo ""

for dir in agent agent/tools bus config cron heartbeat session utils; do
  count=$(find "fubot/$dir" -maxdepth 1 -name "*.py" -exec cat {} + | wc -l)
  printf "  %-16s %5s lines\n" "$dir/" "$count"
done

root=$(cat fubot/__init__.py fubot/__main__.py | wc -l)
printf "  %-16s %5s lines\n" "(root)" "$root"

echo ""
total=$(find fubot -name "*.py" ! -path "*/channels/*" ! -path "*/cli/*" ! -path "*/providers/*" ! -path "*/skills/*" | xargs cat | wc -l)
echo "  Core total:     $total lines"
echo ""
echo "  (excludes: channels/, cli/, providers/, skills/)"
