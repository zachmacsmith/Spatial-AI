#!/bin/bash
# Convenience script to run tests from root directory

echo "Running unit tests..."
conda run -n egoenv python docs/testing/test_comprehensive.py

echo ""
echo "Running end-to-end test..."
conda run -n egoenv python docs/testing/test_end_to_end.py

echo ""
echo "Running preset tests..."
conda run -n egoenv python docs/testing/test_all_presets.py
