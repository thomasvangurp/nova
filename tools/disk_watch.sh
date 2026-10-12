#!/usr/bin/env bash
# Read-only disk watcher (master): logs free space every 10 s; when free drops > 4 GB within ~30 s or falls under
# 15 GB it records what is writing (top writers by /proc/<pid>/io, biggest recently modified files, tmp dirs).
LOG=/root/artifacts/disk_watch.log
prev=$(df -B1 / | tail -1 | awk '{print $4}')
echo "$(date -Is) disk_watch starting free_gb=$(echo "$prev" | awk '{printf "%.1f",$1/1e9}')" >> "$LOG"
tick=0
while true; do
  sleep 10; tick=$((tick+1))
  free=$(df -B1 / | tail -1 | awk '{print $4}')
  drop=$(( (prev - free) / 1000000000 ))
  if [ "$drop" -ge 4 ] || [ "$free" -lt 15000000000 ]; then
    echo "$(date -Is) DISK_EVENT free_gb=$(echo "$free" | awk '{printf "%.1f",$1/1e9}') drop_gb=$drop" >> "$LOG"
    for p in $(ls /proc | grep -E '^[0-9]+$'); do w=$(awk '/^write_bytes/{print $2}' /proc/$p/io 2>/dev/null); [ -n "$w" ] && echo "$p $w"; done > /tmp/dw1.$$
    sleep 5
    for p in $(ls /proc | grep -E '^[0-9]+$'); do w=$(awk '/^write_bytes/{print $2}' /proc/$p/io 2>/dev/null); [ -n "$w" ] && echo "$p $w"; done > /tmp/dw2.$$
    join /tmp/dw1.$$ /tmp/dw2.$$ | awk '{d=$3-$2; if(d>20000000) print d/5/1e6" MB/s pid "$1}' | sort -rn | head -4 | while read a b c d e; do echo "   writer $a MB/s pid $e $(ps -o args= -p $e | cut -c1-110)" >> "$LOG"; done
    rm -f /tmp/dw1.$$ /tmp/dw2.$$
    find /tmp /var/tmp /root/artifacts /root/scratch /root/live -xdev -type f -size +300M -mmin -15 -printf '   bigfile %s %p\n' 2>/dev/null | head -8 >> "$LOG"
    du -sh /tmp /var/tmp 2>/dev/null | sed 's/^/   dir /' >> "$LOG"
    free=$(df -B1 / | tail -1 | awk '{print $4}')
  fi
  prev=$free
  [ $((tick % 360)) -eq 0 ] && echo "$(date -Is) heartbeat free_gb=$(echo "$free" | awk '{printf "%.1f",$1/1e9}')" >> "$LOG"
done
