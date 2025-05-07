#!/usr/bin/env bash
set -e

# Execute the python runner module (gpu_probe.runner), passing along any arguments
# With PYTHONPATH=/app/src, 'gpu_probe' is the top-level package for the module runner.
exec python -m gpu_probe.runner "$@"