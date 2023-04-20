FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Utilities.
RUN apt update ; apt install -y nano git cmake wget htop software-properties-common curl build-essential dpkg

# SSH -- fixing problem in determined ai.
RUN apt-get update && apt-get install -y openssh-server apache2 supervisor
RUN mkdir -p /var/lock/apache2 /var/run/apache2 /var/run/sshd /var/log/supervisor

# Display.
RUN apt update && apt install -y ffmpeg libsm6 libxext6

# Virtual screen.
RUN apt update && apt install -y xvfb xserver-xephyr vnc4server
RUN pip install pyvirtualdisplay

WORKDIR /workspace
RUN git clone https://github.com/anthonysimeonov/relational_ndf.git
WORKDIR /workspace/relational_ndf
RUN pip install numpy cython
RUN pip install -e .
