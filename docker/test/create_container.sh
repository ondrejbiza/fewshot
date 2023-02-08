sudo docker run --net=host --env="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" -v="$HOME/Research/code:/workspace/code:rw" --name=test_pb --gpus device=0 -itd test_pb
