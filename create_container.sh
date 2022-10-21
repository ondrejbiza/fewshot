sudo docker run --net=host --env="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" -v="$HOME/Research/code:/workspace/code:rw" --name=fewshot --gpus device=0 -itd fewshot
