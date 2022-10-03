FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
RUN apt update
# Utilities.
RUN apt install -y nano git cmake wget

# Python packages.
RUN pip install pybullet

# OMPL (https://github.com/ompl)
RUN apt install -y libboost-all-dev
RUN apt install -y libeigen3-dev
# OMPL needs a very particular version of castxml ¯\_(ツ)_/¯
RUN wget https://data.kitware.com/api/v1/item/5f6c9d2450a41e3d19a8e7e8/download
RUN tar xf download
RUN rm download
ENV PATH /workspace/castxml/bin:${PATH}
RUN pip install pygccxml pyplusplus
RUN git clone https://github.com/ompl/ompl.git
RUN mkdir -p /workspace/ompl/build/Release
WORKDIR /workspace/ompl/build/Release
RUN cmake ../.. -DPYTHON_EXEC=/opt/conda/bin/python
RUN make -j 8 update_bindings
RUN make -j 8
WORKDIR /workspace

# Allow pybullet to connect to the host's display.
# docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -it image
RUN apt install -y libglu1-mesa-dev libgl1-mesa-dri freeglut3-dev mesa-common-dev
RUN apt install -y libnvidia-gl-515
RUN apt install -y x11-apps