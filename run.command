#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import numpy,pandas,scipy,torch' >/dev/null 2>&1; then
  .venv/bin/python run.py "$@"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import numpy,pandas,scipy,torch' >/dev/null 2>&1; then
  python3 run.py "$@"
elif [ -x "$HOME/anaconda3/bin/python" ] && "$HOME/anaconda3/bin/python" -c 'import numpy,pandas,scipy,torch' >/dev/null 2>&1; then
  "$HOME/anaconda3/bin/python" run.py "$@"
else
  echo 'Install requirements.txt in a Python environment first. See README.md.'
fi
read -r -p 'Press Enter to close...'
