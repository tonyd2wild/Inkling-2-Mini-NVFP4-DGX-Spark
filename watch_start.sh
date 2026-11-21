#!/usr/bin/env bash
# Watch an Inkling profile while it starts (GB10 unified-memory safety net).
# Every 5 s: logs MemAvailable (MiB), the lowest seen, container state and the last log line.
# Kills the container if MemAvailable drops under FLOOR_KILL_GIB (default 3) so the box never hangs.
# Exit 0 = ready, 2 = killed on memory floor, 3 = container exited, 4 = timeout (15 min).
# usage: watch_start.sh [logfile]      (default ./logs/inkling_start_watch.log)
NAME="${NAME:-inkling2mini}"
FLOOR_KILL_MB=$(( ${FLOOR_KILL_GIB:-3} * 1024 ))
LOG="${1:-./logs/inkling_start_watch.log}"
mkdir -p "$(dirname "$LOG")"
min=999999
for i in $(seq 1 180); do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  [ "$avail" -lt "$min" ] && min=$avail
  state=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo gone)
  stage=$(docker logs --tail 1 "$NAME" 2>&1 | tail -c 160 | tr -d '\r')
  echo "$(date -u +%T) avail_MiB=$avail min_MiB=$min state=$state | $stage" >> "$LOG"
  if [ "$avail" -lt "$FLOOR_KILL_MB" ]; then
    echo "KILL: MemAvailable ${avail} MiB under floor ${FLOOR_KILL_MB} MiB" >> "$LOG"
    docker rm -f "$NAME" >/dev/null 2>&1
    exit 2
  fi
  if [ "$state" != "running" ]; then
    echo "EXITED: state=$state" >> "$LOG"
    exit 3
  fi
  if docker logs "$NAME" 2>&1 | grep -q "Application startup complete"; then
    echo "READY min_MiB=$min" >> "$LOG"
    exit 0
  fi
  sleep 5
done
echo "TIMEOUT" >> "$LOG"
exit 4
