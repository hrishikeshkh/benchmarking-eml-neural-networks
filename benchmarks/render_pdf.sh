#!/bin/sh
# Render paper/index.html to paper/paper.pdf with a headless Chromium (MathJax needs network access).
set -e
CHROME="${CHROME:-$(ls -d ~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell 2>/dev/null | tail -1)}"
cd "$(dirname "$0")/../paper"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --virtual-time-budget=20000 \
  --print-to-pdf="$(pwd)/paper.pdf" "file://$(pwd)/index.html"
