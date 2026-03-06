#!/usr/bin/env bash

# ============================================================
# SIPA Demo Runner
# ------------------------------------------------------------
# Runs two benchmark trajectories:
#   1. Physically consistent trajectory
#   2. Corrupted trajectory (physical hallucination)
#
# Generates diagnostic PIR plots for comparison.
# ============================================================

set -e

echo ""
echo "==============================================="
echo "SIPA Demo — Physical Consistency Audit"
echo "==============================================="
echo ""

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

DEMO_DIR="$ROOT_DIR/demo"
OUTPUT_DIR="$ROOT_DIR/outputs"

mkdir -p "$OUTPUT_DIR"

echo "[1/2] Running NORMAL trajectory audit..."
python "$ROOT_DIR/scripts/run_audit.py" \
  --input "$DEMO_DIR/sipa_minimal_trajectory.csv" \
  --dt 0.01 \
  --branding

echo ""
echo "[2/2] Running CORRUPTED trajectory audit..."
python "$ROOT_DIR/scripts/run_audit.py" \
  --input "$DEMO_DIR/sipa_corrupted_trajectory.csv" \
  --dt 0.01 \
  --branding

echo ""
echo "==============================================="
echo "Demo completed."
echo ""
echo "Output files generated in:"
echo "   outputs/"
echo ""
echo "Look for:"
echo "   sipa_audit_pir_evolution.png"
echo ""
echo "==============================================="
