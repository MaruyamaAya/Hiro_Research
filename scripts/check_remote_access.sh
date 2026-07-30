#!/usr/bin/env bash
set -u

if [[ $# -gt 0 ]]; then
    HOSTS=("$@")
else
    HOSTS=(stevenzmliu-any4 stevenzmliu-any2)
fi
failed=0
for host in "${HOSTS[@]}"; do
    echo "=== $host ==="
    resolved="$(ssh -G "$host" 2>/dev/null | awk '$1 == "hostname" {print $2; exit}')"
    port="$(ssh -G "$host" 2>/dev/null | awk '$1 == "port" {print $2; exit}')"
    echo "endpoint=${resolved:-unknown}:${port:-unknown}"
    if nc -z -G 5 "$resolved" "$port" 2>/dev/null; then
        echo "tcp=reachable"
    else
        echo "tcp=unreachable"
        failed=1
        continue
    fi
    output="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" 'printf connected; hostname' 2>&1)"
    status=$?
    if [[ $status -eq 0 ]]; then
        echo "ssh=ready"
        echo "$output"
    else
        echo "ssh=failed"
        echo "$output"
        failed=1
    fi
    echo
done
exit "$failed"
