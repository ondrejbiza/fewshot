FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Utilities.
RUN apt update ; apt install -y nano git cmake wget htop software-properties-common curl build-essential dpkg

# 'tzdata' have an annoying prompt.
RUN apt update ; DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt install -y tzdata

# Compile Python 2.7.18.
RUN apt update ; apt install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
RUN wget https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
RUN tar xzf Python-2.7.18.tgz
WORKDIR /workspace/Python-2.7.18
RUN ./configure --enable-optimizations
RUN make altinstall
WORKDIR /workspace

# Install Neural Descriptor Fields.
RUN git clone --recursive https://github.com/anthonysimeonov/ndf_robot.git
WORKDIR /workspace/ndf_robot
RUN pip install -e .
WORKDIR /workspace//ndf_robot/pybullet-planning/pybullet_tools/ikfast/franka_panda
RUN python setup.py
WORKDIR /workspace/ndf_robot
# RUN ./scripts/download_demo_data.sh
# RUN ./scripts/download_demo_demonstrations.sh
# RUN ./scripts/download_demo_weights.sh
# RUN ./scripts/download_obj_data.sh
