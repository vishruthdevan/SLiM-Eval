#!/bin/bash

echo "=========================================="
echo "LLM Evaluation Setup"
echo "=========================================="

# Activate conda environment
echo "Activating conda environment 'slimeval'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate slimeval

# Check if activation succeeded
if [ "$CONDA_DEFAULT_ENV" != "slimeval" ]; then
    echo "❌ Failed to activate conda environment 'slimeval'"
    echo "Please make sure the environment exists:"
    echo "  conda create -n slimeval python=3.10"
    exit 1
fi

echo "✓ Environment activated: $CONDA_DEFAULT_ENV"

# Install/update dependencies
echo ""
echo "Checking dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Enable HumanEval code execution
export HF_ALLOW_CODE_EVAL="1"

# Note about sudo access
echo ""
echo "=========================================="
echo "⚠️  IMPORTANT: Energy Monitoring"
echo "=========================================="
echo "Energy monitoring requires sudo access for powermetrics."
echo "You will be prompted for your password when the script runs."
echo "Press Ctrl+C now if you need to set this up first."
echo ""
read -p "Press Enter to continue..."

# Run evaluation
echo ""
echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="
python evaluate_model.py

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Check the results/ directory for output files."