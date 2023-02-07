FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Utilities.
RUN apt update ; apt install -y nano git cmake wget htop software-properties-common curl build-essential dpkg

# 'tzdata' have an annoying prompt.
RUN apt update ; DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt install -y tzdata

# Install ROS Noetic.
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt update ; apt install ros-noetic-desktop-full
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN source /opt/ros/noetic/setup.bash ; rosdep init
RUN source /opt/ros/noetic/setup.bash ; rosdep update
RUN apt update ; apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN apt update ; apt install -y ros-noetic-moveit-kinematics
