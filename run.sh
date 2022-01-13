#!/bin/sh

echo ' ####### start to train ######## '

if [ $1 = "debug" ]; then
        python exp/exp005.py \
                --debug \
                --train_fold 0
fi

if [ $1 = "all" ]; then
        python exp/ex005.py \
                --train_fold 0 1 2 3 4 5 6 7 8 9
fi

if [ $1 = "val" ]; then
        python exp/exp005.py \
                --train_fold 0
fi