#!/usr/bin/env bash
# raport.html -> raport.pdf (A4) przez headless Chrome; kontrola liczby stron.
set -euo pipefail
cd "$(dirname "$0")"
google-chrome --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/raport.pdf" "file://$PWD/raport.html" 2>/dev/null
pdfinfo raport.pdf | grep -E "Pages|Page size"
