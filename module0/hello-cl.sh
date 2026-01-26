#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

OUT="hello_opencl_output.txt"

# Start fresh
echo "OpenCL HelloWorld Execution Log" > "$OUT"
echo "===============================" >> "$OUT"
echo "" >> "$OUT"

run() {
  echo "\$ $*" >> "$OUT"
  "$@" >> "$OUT" 2>&1
  echo "" >> "$OUT"
}

# Optional but recommended: show OpenCL devices
# (clinfo may not be installed everywhere)
if command -v clinfo >/dev/null 2>&1; then
  run clinfo
else
  echo "clinfo not installed; skipping device query" >> "$OUT"
  echo "" >> "$OUT"
fi

# Show source
run cat hello_world_cl.c

# Compile (adjust library name if needed on your system)
run gcc hello_world_cl.c -o hello_cl -lOpenCL

# Run
run ./hello_cl
