#!/bin/bash

# Script to run a sweep locally with test modifications
# Usage: ./run_sweep_local_test.sh <sweep-config-file>
# Usage: ./run_sweep_local_test.sh --dry-run <sweep-config-file>

# Check for dry-run flag
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    shift
fi

# Check if help or no arguments
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--dry-run] <sweep-config-file>"
    echo "Example: $0 sweep-configs/my-sweep.yaml"
    echo "Example: $0 --dry-run sweep-configs/my-sweep.yaml"
    echo ""
    echo "This script will modify the sweep config for local testing with:"
    echo "  - batch_size=1"
    echo "  - project=nevermind"
    echo "  - run_cap=1"
    echo "  - trainer.limit_train_batches=5"
    echo "  - trainer.limit_val_batches=2"
    echo "  - trainer.max_epochs=4"
    echo ""
    echo "Options:"
    echo "  --dry-run    Show changes without running the sweep"
    exit 1
fi

SWEEP_CONFIG="$1"

# Check if sweep config file exists
if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Error: Sweep config file '$SWEEP_CONFIG' not found"
    exit 1
fi

echo "Creating local test version of: $SWEEP_CONFIG"

# Create temporary file
TEMP_CONFIG=$(mktemp /tmp/sweep_local_test.XXXXXX.yaml)
trap "rm -f $TEMP_CONFIG" EXIT

# Copy original config to temp file
cp "$SWEEP_CONFIG" "$TEMP_CONFIG"

# Modify the temporary config file for local testing
echo "Modifying config for local testing..."

# 1. Change project to "nevermind"
if grep -q "^project:" "$TEMP_CONFIG"; then
    sed -i 's/^project:.*/project: nevermind/' "$TEMP_CONFIG"
else
    # If project line doesn't exist, add it after program line
    sed -i '/^program:/a\
project: nevermind' "$TEMP_CONFIG"
fi

# 2. Change run_cap to 1
if grep -q "^run_cap:" "$TEMP_CONFIG"; then
    sed -i 's/^run_cap:.*/run_cap: 1/' "$TEMP_CONFIG"
else
    # If run_cap doesn't exist, add it after project line
    sed -i '/^project:/a\
run_cap: 1' "$TEMP_CONFIG"
fi

# 3. Add batch_size parameter
# First, check if loader.batch_size already exists in parameters section
if grep -q "loader\.batch_size:" "$TEMP_CONFIG"; then
    # Replace existing loader.batch_size section with value: 1
    # Use awk to handle the replacement more reliably
    awk '
    /^  loader\.batch_size:/ {
        print "  loader.batch_size:"
        print "    value: 1"
        skip = 1
        next
    }
    skip && /^  [a-zA-Z_]/ {
        skip = 0
    }
    !skip {
        print
    }
    ' "$TEMP_CONFIG" > "$TEMP_CONFIG.tmp" && mv "$TEMP_CONFIG.tmp" "$TEMP_CONFIG"
elif grep -q "^parameters:" "$TEMP_CONFIG"; then
    # Add loader.batch_size under parameters section
    sed -i '/^parameters:/a\
  loader.batch_size:\
    value: 1' "$TEMP_CONFIG"
else
    # If parameters section doesn't exist, add it at the end
    echo "" >> "$TEMP_CONFIG"
    echo "parameters:" >> "$TEMP_CONFIG"
    echo "  loader.batch_size:" >> "$TEMP_CONFIG"
    echo "    value: 1" >> "$TEMP_CONFIG"
fi

# 4. Add trainer parameters after ${args_no_hyphens} line
if grep -q "\${args_no_hyphens}" "$TEMP_CONFIG"; then
    # Add trainer parameters after the ${args_no_hyphens} line
    sed -i '/\${args_no_hyphens}/a\
  - trainer.limit_train_batches=5\
  - trainer.limit_val_batches=2\
  - trainer.max_epochs=4' "$TEMP_CONFIG"
else
    echo "Warning: ${args_no_hyphens} line not found in config file"
fi

echo "Modified config saved to: $TEMP_CONFIG"
echo ""
echo "Changes made:"
echo "  ✓ project: nevermind"
echo "  ✓ run_cap: 1" 
echo "  ✓ loader.batch_size: 1"
echo "  ✓ trainer.limit_train_batches: 5"
echo "  ✓ trainer.limit_val_batches: 2"
echo "  ✓ trainer.max_epochs: 4"
echo ""

# Show the diff if possible
if command -v diff >/dev/null 2>&1; then
    echo "Diff of changes:"
    echo "=================="
    diff "$SWEEP_CONFIG" "$TEMP_CONFIG" || true
    echo ""
fi

# Ask for confirmation
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN: Would run sweep with this modified config."
    echo "Temporary config content:"
    echo "=========================="
    cat "$TEMP_CONFIG"
    exit 0
fi

read -p "Proceed with running the local test sweep? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# echo "The tmp sweep config"
# cat 

echo "Running local test sweep..."
echo "=========================="

# Run the sweep using run_sweep.sh with the temporary config
./run_sweep.sh "$TEMP_CONFIG"
