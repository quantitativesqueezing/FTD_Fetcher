#!/bin/bash

# Prompt user for number of results
while true; do
    read -p "How many top FTD results do you want to export? (must be > 0): " num_results
    if [[ $num_results =~ ^[1-9][0-9]*$ ]]; then
        break
    else
        echo "Error: Please enter a valid integer greater than 0."
    fi
done

echo "Fetching top $num_results FTD results..."

# Activate Virtual Environment
python3 -m venv ./venv
source ./venv/bin/activate

# Install Dependencies
python3 -m pip install -U requests pandas openpyxl
python3 -m pip install -U git+https://github.com/Pycord-Development/pycord

# Run the Script with the specified number of results.
python3 FTD_Fetcher.py $num_results

# Remove the Virtual Environment when we're finished
rm -rf ./venv