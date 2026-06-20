#!/bin/bash

docker run -it \
  --gpus all \
  --device=/dev/video0:/dev/video0 \
  -p 8080:8080 \
  -p 7860:7860 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/projects/PersonaLive:/workspaces/PersonaLive \
  -v $HOME/projects/LHM:/workspaces/LHM \
  -w /workspaces \
  cobaltconcrete/lhm-personalive:cu121-torch230 \
  /bin/bash