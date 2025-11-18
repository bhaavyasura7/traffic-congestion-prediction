#!/bin/bash

# This script runs the entire traffic analysis pipeline in the correct order.
# The 'set -e' command ensures that the script will exit immediately if any command fails.
set -e

echo "--- [START] Traffic Analysis Pipeline ---"
echo ""

echo "--- [Step 1/5] Processing Video and Counting Vehicles ---"
python test2.py
echo ""
echo "✅ Step 1 complete."
echo ""

echo "--- [Step 2/5] Performing Clustering on Vehicle Data ---"
echo ""
echo "✅ Step 2 complete."
echo ""

echo "--- [Step 3/5] Training Prediction Model ---"
python phase3_prediction.py
echo ""
echo "✅ Step 3 complete."
echo ""

echo "--- [Step 4/5] Predicting Future Hourly Congestion ---"
python congestion_predictor.py
echo ""
echo "✅ Step 4 complete."
echo ""


echo "--- [SUCCESS] Pipeline finished successfully! ---"