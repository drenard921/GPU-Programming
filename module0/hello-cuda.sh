#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

OUT="hello_cuda_output.txt"

# Start fresh
echo "CUDA HelloWorld Execution Log" > "$OUT"
echo "=============================" >> "$OUT"
echo "" >> "$OUT"

run() {
  echo "\$ $*" >> "$OUT"
  "$@" >> "$OUT" 2>&1
  echo "" >> "$OUT"
}



run nvidia-smi
run nvcc --version

run cat hello-world.cu 

run nvcc -o hello hello-world.cu
run ./hello