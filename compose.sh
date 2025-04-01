
# usage: 
# sudo bash -c "source compose.sh"
# from this directory

# export DISPLAY=localhost:11.0


docker build -t lns2rl:latest -f Dockerfile .
docker run -dit \
    --name lns2rlros \
    --network host \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /root/.Xauthority:/root/.Xauthority \
    -v /data:/data \
    -v /home/ubuntu/LNS2-RL:/lns2rl \
    -p 5005:5005 \
    lns2rl:latest
