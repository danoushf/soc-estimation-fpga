#!/bin/bash

# GitHub Upload Script for Battery SoC Estimation Project
# This script will help you upload your project to GitHub

echo "Battery SoC Estimation - GitHub Upload Script"
echo "=============================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git first."
    exit 1
fi

# Navigate to project directory
cd /home/dna/battery-state-estimation/battery-state-estimation

# Initialize git repository (if not already initialized)
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
fi

# Add all files to staging
echo "Adding files to Git staging..."
git add .

# Commit files
echo "Committing files..."
git commit -m "Initial commit: Battery SoC Estimation with Deep Learning

- Added comprehensive README with project overview and features
- Implemented multiple model architectures (LSTM, Bi-LSTM, GRU, Bi-GRU, 1D CNN)
- Included Bayesian hyperparameter optimization
- Added sliding window preprocessing for time series data
- Created requirements.txt with all dependencies
- Added setup guide and documentation
- Included proper project structure with organized directories
- Added evaluation metrics and visualization capabilities
- Designed for FPGA implementation considerations"

# Add remote repository (replace with your actual repository URL)
echo "Adding remote repository..."
git remote add origin https://github.com/danoushf/soc-estimation-fpga.git

# Push to GitHub
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "=============================================="
echo "Upload completed successfully!"
echo "Your project is now available at:"
echo "https://github.com/danoushf/soc-estimation-fpga"
echo "=============================================="
