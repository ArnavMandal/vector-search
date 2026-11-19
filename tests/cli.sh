#!/bin/sh
set -eu
cli=$1
actual=$("$cli" examples/points.txt '0 0')
expected=$(printf 'id\tsquared_l2\n1\t0\n2\t1\n3\t1\n4\t1\n5\t4\n6\t4\n7\t8\n8\t8\n9\t9\n10\t9\n')
[ "$actual" = "$expected" ]
for bad_query in '1' '0 0 0' 'nan 1' '2x 0' ''; do
    if "$cli" examples/points.txt "$bad_query" >/dev/null 2>&1; then exit 1; fi
done
if "$cli" examples/points.txt '0 0' 0 >/dev/null 2>&1; then exit 1; fi
echo "CLI top-10 and invalid input tests passed"
