#!/bin/bash
# Desired output:
# Skiprun passed.
# n007 : PASSED
# n015 : PASSED
# n023 : PASSED
# n142 : FAILED  # no gpu

# skiprun at host
source ../setup.sh >/dev/null 2>&1 && \
lmp -skiprun -in lammps.in > log.host 2>&1

ret=$?
if [ $ret -eq 0 ]; then
    echo "Skiprun passed."
else
    echo "Skiprun failed."
fi

# actual run at GPU nodes
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES=(n007 n015 n023 n142)

for node in "${NODES[@]}"; do
(
    if ssh -q "$node" \
       "bash -lc 'cd $WORKDIR && source ../setup.sh >/dev/null 2>&1 && lmp -in lammps.in > log.$node 2>&1'" \
       >/dev/null 2>&1
    then
        echo "$node : PASSED"
    else
        echo "$node : FAILED"
    fi
) &
done

wait

# remove intermediate logs
rm log.* trajectory.dcd
