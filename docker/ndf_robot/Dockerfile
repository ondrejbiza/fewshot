FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Utilities.
RUN apt update ; apt install -y nano git cmake wget htop software-properties-common curl build-essential dpkg

# 'tzdata' have an annoying prompt.
RUN apt update ; DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt install -y tzdata

