#!/usr/bin/env bash

set -e
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/sample_patient.json"
OUTPUT="results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

uv run python -m src.predict \
  --input "$INPUT" \
  --output "$OUTPUT"