sudo docker run --net=host --env="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" -v="$HOME/Research/code:/workspace/code:rw" --name=ndf_robot --gpus device=0 -itd ndf_robot
