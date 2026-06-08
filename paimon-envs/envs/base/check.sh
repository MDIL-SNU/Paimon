#!/bin/bash
# Validates base environment integrity after setup.sh is sourced.
# Usage: source setup.sh && bash check.sh
set -e

# packmol binary
which packmol > /dev/null 2>&1
