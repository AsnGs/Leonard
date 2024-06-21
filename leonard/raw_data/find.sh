#!/bin/bash

# 使用 grep 和 cut 查找指定条件的行，并打印行号
line_number=$(grep -n '"f000"' vertex200m.csv | head -n 1 | cut -d':' -f1)

echo "行号为: $line_number"
