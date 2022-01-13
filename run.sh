#!/bin/sh

echo ' ####### start to train ######## '

if [ $1 != "debug"]; then
        python exp/exp004.py \
                --train-fold 0 1 2 3 4 5 6 7 8 9
else
        python exp/exp004.py \
                --debug \
                --train-fold 0
if
