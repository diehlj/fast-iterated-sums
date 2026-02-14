#!/bin/bash

# Check if sweep config is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <sweep-config-file>"
    echo "Example: $0 sweep-configs/my-sweep.yaml"
    exit 1
fi

SWEEP_CONFIG="$1"

# Check if sweep config file exists
if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Error: Sweep config file '$SWEEP_CONFIG' not found"
    exit 1
fi

echo "Starting wandb sweep with config: $SWEEP_CONFIG"

# Check if torch is available, if not, set up the environment
if ! python -c "import torch" 2>/dev/null; then
    echo "torch not available, setting up environment..."
    exit 1
    # XXX This only works on brain.
    #source ~/.bashrc
    #conda activate fis-review-py312
    #echo "using python:"
    #which python
else
    echo "torch is available, using current environment"
    echo "using python:"
    which python
fi

# Create temporary file for stderr
STDERR_FILE=$(mktemp /tmp/wandb_sweep_stderr.XXXXXX)
trap "rm -f $STDERR_FILE" EXIT

# Run wandb sweep and capture stderr
echo "Running: wandb sweep $SWEEP_CONFIG"
wandb sweep "$SWEEP_CONFIG" 2> >(tee "$STDERR_FILE" >&2)

# Parse the sweep agent command from stderr
AGENT_COMMAND=$(grep "wandb agent" "$STDERR_FILE" | sed 's/.*wandb: Run sweep agent with: //' | head -1)

if [ -z "$AGENT_COMMAND" ]; then
    echo "Error: Could not find 'wandb agent' command in sweep output"
    echo "Stderr content:"
    cat "$STDERR_FILE"
    exit 1
fi

echo ""
echo "Found agent command: $AGENT_COMMAND"
echo "Starting sweep agent..."
echo ""

# Run the agent command
$AGENT_COMMAND
