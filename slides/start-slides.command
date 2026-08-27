#!/bin/zsh
set -e
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"
/usr/bin/python3 "$SCRIPT_DIR/serve_slides.py" --host 127.0.0.1 --port 8765 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM
for _ in {1..40}; do
  if /usr/bin/curl -fsS "http://127.0.0.1:8765/api/feedback" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
open "http://127.0.0.1:8765/slides.html"
wait "$SERVER_PID"
