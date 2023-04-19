FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Utilities.
RUN apt update ; apt install -y nano git cmake wget htop software-properties-common curl build-essential dpkg

# Display.
RUN apt update && apt install -y ffmpeg libsm6 libxext6

WORKDIR /workspace
RUN git clone https://github.com/anthonysimeonov/relational_ndf.git
WORKDIR /workspace/relational_ndf
RUN pip install numpy cython
RUN pip install -e .
