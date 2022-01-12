#!/bin/sh
[ ! -d "~/kaggle-image" ] && git clone https://github.com/Kaggle/docker-python.git ~/kaggle-image

cd ~/kaggle-image 
./build --gpu --use-cache
cd -

