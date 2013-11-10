#!/bin/bash

export MAXENTPATH=(/home/iiotep9huy/Templates/github/maxent/python/build/lib.*)
export PYTHONPATH=${MAXENTPATH}:${PYTHONPATH}
echo "=== PYTHONPATH=${PYTHONPATH}" >&2

exec python2 ./run.py "$@"
