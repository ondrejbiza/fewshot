FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Utilities.
RUN apt update ; apt install -y nano git cmake wget htop software-properties-common curl build-essential dpkg

# 'tzdata' have an annoying prompt.
RUN apt update ; DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt install -y tzdata

# Allow pybullet to connect to the host's display.
# Run `xhost +` on the host machine.
# docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -it image
RUN apt install -y libglu1-mesa-dev libgl1-mesa-dri freeglut3-dev mesa-common-dev
# Insert your nvidia driver version here.
RUN apt install -y libnvidia-gl-510
RUN apt install -y x11-apps
