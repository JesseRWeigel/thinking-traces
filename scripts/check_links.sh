#!/usr/bin/env bash
# Resolve every outbound link in the published page and the README against the
# real internet, and fetch the deployed page itself.
#
# Kept out of scripts/verify.sh on purpose: verify has to be deterministic and
# offline, and a network check is neither. Run this before publishing and paste
# the result into the README. A confidently written repo URL that has never
# existed looks correct in review and 404s in public.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SITE="${1:-https://jesserweigel.github.io/thinking-traces/}"
FAILED=0

echo "fetching the deployed page: $SITE"
BODY="$(curl -sSL --max-time 30 "$SITE")" || { echo "  FAIL: could not fetch"; FAILED=1; }
if [ -n "${BODY:-}" ]; then
  if grep -q "What thinking traces actually buy you" <<<"$BODY"; then
    echo "  ok: deployed page carries the expected title"
  else
    echo "  FAIL: deployed page does not look like this project"
    FAILED=1
  fi
  DIGITS=$(grep -o '[0-9]' <<<"$BODY" | wc -l)
  if [ "$DIGITS" -lt 200 ]; then
    echo "  FAIL: deployed page has only $DIGITS digits, it is not showing results"
    FAILED=1
  else
    echo "  ok: deployed page carries $DIGITS digits of results"
  fi
fi

echo "checking outbound links"
LINKS=$(grep -ohE 'https?://[^"<> )]+' docs/index.html README.md | sed 's/[.,]$//' | sort -u)
for url in $LINKS; do
  code=$(curl -sSL -o /dev/null -w '%{http_code}' --max-time 30 "$url")
  if [ "$code" = "200" ]; then
    echo "  ok  $code  $url"
  else
    echo "  FAIL $code  $url"
    FAILED=1
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo "LINK CHECK FAILED"
  exit 1
fi
echo "LINK CHECK OK"
