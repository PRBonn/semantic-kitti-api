#!/bin/bash
# This file is covered by the LICENSE file in the root of this project.
docker build -t api --build-arg uid=$(id -g) --build-arg gid=$(id -g) .
docker run --privileged \
       -ti --rm -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v $1:/home/developer/data/ \
       api