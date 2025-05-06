#!/usr/bin/env bash
set -e

# Execute the python runner module (gpu_probe.runner), passing along any arguments
exec python -m gpu_probe.runner "$@"