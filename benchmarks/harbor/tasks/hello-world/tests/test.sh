#!/bin/bash
# Verification: check that /tmp/hello.sh exists and outputs "hello world"
set -euo pipefail
if [ ! -f /tmp/hello.sh ]; then
    echo "FAIL: /tmp/hello.sh not found"
    exit 1
fi
output=$(bash /tmp/hello.sh 2>&1)
if echo "$output" | grep -qi "hello world"; then
    echo "PASS: $output"
else
    echo "FAIL: expected 'hello world', got '$output'"
    exit 1
fi
