#!/bin/bash

if [[ $1 == 1 ]]; then
    python q1.py $2 $3 $4 $5
elif [[ $1 == 2 ]]; then
    python q2.py $2 $3 $4
elif [[ $1 == 3 ]]; then
    python q3.py $2 $3 
elif [[ $1 == 4 ]]; then
    python q4.py $2 $3 $4
fi