#!/usr/bin/env bash
# Checks every 30 seconds while the market window is open, inside ONE GitHub Actions job.
# GitHub's own schedules skip or delay runs on busy days (Oct 20 2026: the first start came 4h48m late), so the loop
# never relies on them: it stays alive around the clock, idling outside market hours, and near GitHub's 6-hour job cap
# it starts the next run of this workflow itself. The schedule is only a backup in case that chain ever breaks.
set -u
start=$(date +%s)
git config --global user.name "paper-bot"
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"

in_window() {  # weekdays 13:00-21:10 UTC covers 9:00am-4:00pm New York time in both EDT and EST
  if [ -n "${TEST_MINUTES:-}" ]; then return 0; fi
  local dow hm
  dow=$(date -u +%u); hm=$(date -u +%H%M)
  [ "$dow" -le 5 ] && [ "$((10#$hm))" -ge 1300 ] && [ "$((10#$hm))" -le 2110 ]
}

publish() {  # one commit, force-pushed, so the data branch never grows
  rm -rf /tmp/data && mkdir -p /tmp/data && cp docs/scanner.json /tmp/data/
  (cd /tmp/data && git init -q -b data && git add -A     && git commit -q -m "live data $(date -u +'%Y-%m-%d %H:%M:%S') UTC"     && git push -q -f "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" data)     || echo "publish failed (will retry next round)"
}

last_save=0
limit=$((330 * 60))   # hand over at 5.5 hours, before GitHub's 6-hour cap
[ -n "${TEST_MINUTES:-}" ] && limit=$((TEST_MINUTES * 60))
while true; do
  if in_window; then
    git pull -q --rebase --autostash || true
    python scanner.py --tick || echo "tick failed (will retry next round)"
    # save to main when a trade or scan happened, otherwise every 15 minutes (the dashboard gets every check anyway)
    if [ -f .changed ] || [ $(( $(date +%s) - last_save )) -ge 900 ]; then
      rm -f .changed
      last_save=$(date +%s)
      git add scanner_state.json
    fi
    if ! git diff --cached --quiet; then
      git commit -q -m "bot: $(date -u +'%Y-%m-%d %H:%M') UTC"
      git pull -q --rebase --autostash && git push -q || echo "push failed (will retry next round)"
    fi
    publish
  fi
  if [ $(( $(date +%s) - start )) -gt "$limit" ]; then
    git add scanner_state.json   # hand the latest account to the next run
    if ! git diff --cached --quiet; then
      git commit -q -m "bot: $(date -u +'%Y-%m-%d %H:%M') UTC (handover)"
      git pull -q --rebase --autostash && git push -q || echo "final push failed"
    fi
    if [ -z "${TEST_MINUTES:-}" ]; then
      echo "near the 6-hour job limit: starting the next run to keep watching"
      gh workflow run trader.yml --ref main || echo "could not start the next run; the backup schedule will"
    fi
    break
  fi
  if in_window; then
    sleep $(( 30 - $(date +%s) % 30 + 2 ))   # wake 2 seconds after the next :00 or :30
  else
    sleep 60                                  # market closed: idle, check the clock once a minute
  fi
done
echo "loop finished at $(date -u +%H:%M) UTC"
