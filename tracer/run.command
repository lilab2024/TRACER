#!/bin/zsh
cd "${0:A:h}"
if command -v python >/dev/null 2>&1; then
    python run.py
else
    python3 run.py
fi
run_exit_code=$?
echo "Exit status: $run_exit_code"
read -r '?Press Enter to close...'
exit $run_exit_code
