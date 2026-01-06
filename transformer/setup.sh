#!/bin/bash

echo "========================================="
echo "Transformer Project Setup"
echo "========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "Creating directories..."
mkdir -p data checkpoints logs

# Run example setup
echo "Setting up example dataset..."
python setup_example.py

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To train on example data:"
echo "  python train.py --config config_small.json"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
