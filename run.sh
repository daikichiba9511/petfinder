#!/bin/sh

echo ' ####### start to train ######## '

python exp/exp003.py \
        # --debug \
        # --train-fold 0
        --train-fold 0 1 2 3 4 5 6 7 8 9
