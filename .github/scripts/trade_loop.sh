#!/usr/bin/env bash
# Checks every 1 minute for as long as the market window is open, inside ONE GitHub Actions job.
# GitHub's own schedules skip runs on busy days (Oct 19 2026: 2 runs instead of ~100), so we don't rely on them for
# the 5-minute rhythm, only to start this loop. Jobs are capped at 6 hours, so near that limit the loop starts a
# fresh run of this workflow to carry on.
set -u
start=$(date +%s)
git config --global user.name "paper-bot"
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"

in_window() {  # weekdays 13:00-21:10 UTC covers 9:00am-4:00pm New York time in both EDT and EST
  if [ -n "${TEST_MINUTES:-}" ]; then [ $(( $(date +%s) - start )) -lt $(( TEST_MINUTES * 60 )) ]; return; fi
  local dow hm
  dow=$(date -u +%u); hm=$(date -u +%H%M)
  [ "$dow" -le 5 ] && [ "$((10#$hm))" -ge 1300 ] && [ "$((10#$hm))" -le 2110 ]
}

publish() {  # one commit, force-pushed, so the data branch never grows
  rm -rf /tmp/data && mkdir -p /tmp/data && cp docs/scanner.json /tmp/data/
  (cd /tmp/data && git init -q -b data && git add -A     && git commit -q -m "live data $(date -u +'%Y-%m-%d %H:%M') UTC"     && git push -q -f "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" data)     || echo "publish failed (will retry next round)"
}

last_save=0
limit=$((330 * 60))   # stop at 5.5 hours, before GitHub's 6-hour cap
while in_window; do
  python scanner.py --tick || echo "tick failed (will retry next round)"
  # save to main when a trade or scan happened, otherwise every 15 minutes (the dashboard gets every minute anyway)
  if [ -f .changed ] || [ $(( $(date +%s) - last_save )) -ge 900 ]; then
    rm -f .changed
    last_save=$(date +%s)
    git add scanner_state.json
  fi
  if ! git diff --cached --quiet; then
    git commit -q -m "bot: $(date -u +'%Y-%m-%d %H:%M') UTC"
    git pull -q --rebase && git push -q || echo "push failed (will retry next round)"
  fi
  publish
  if [ $(( $(date +%s) - start )) -gt "$limit" ]; then
    if in_window; then
      echo "near the 6-hour job limit: starting a fresh run to keep trading"
      gh workflow run trader.yml --ref main || echo "could not start the next run; the backup schedule will"
    fi
    break
  fi
  sleep $(( 60 - $(date +%s) % 60 + 5 ))   # wake 5 seconds after the next minute mark
done
echo "loop finished at $(date -u +%H:%M) UTC"
