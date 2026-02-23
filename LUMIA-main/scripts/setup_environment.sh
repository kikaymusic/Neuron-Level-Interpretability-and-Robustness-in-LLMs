#!/bin/bash

# Script to set up the Lumia environment

# Step 1: Create a new conda environment
echo "Creating new conda environment 'lumia'..."
conda create -n lumia python=3.10.14 -y

if [ $? -ne 0 ]; then
    echo "Failed to create the conda environment. Exiting."
    exit 1
fi

# Activate the environment
echo "Activating the 'lumia' environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lumia

if [ $? -ne 0 ]; then
    echo "Failed to activate the conda environment. Exiting."
    exit 1
fi

# Step 2: Install requirements
if [ -f requirements.txt ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo "Failed to install requirements. Exiting."
        exit 1
    fi
else
    echo "requirements.txt not found. Skipping requirements installation."
fi



echo "Setup completed successfully!"
