#!/bin/bash
filename=$1

while read line; do
echo "installing $line"; 
pip install line;
done < $filename
