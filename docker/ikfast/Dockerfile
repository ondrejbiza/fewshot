FROM library/ubuntu:14.04

# Utilities.
RUN apt-get update ; apt-get install -y nano git cmake wget htop software-properties-common curl build-essential dpkg

# 'tzdata' have an annoying prompt.
RUN apt-get update ; DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt install -y tzdata

# Install ROS Indigo.
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt-cache search ros-indigo
# RUN apt-get update ; apt-get install ros-indigo-desktop-full
# RUN echo "source /opt/ros/indigo/setup.bash" >> /root/.bashrc
# RUN source /opt/ros/noetic/setup.bash ; rosdep init
# RUN source /opt/ros/noetic/setup.bash ; rosdep update
# RUN apt-get update ; apt-get install python-rosinstall
# RUN apt-get update ; apt-get install ros-indigo-moveit-kinematics

# # Install OpenRAVE prerequisites.
# RUN apt-get install cmake g++ git ipython minizip python-dev python-h5py python-numpy python-scipy qt4-dev-tools
# RUN apt-get install libassimp-dev libavcodec-dev libavformat-dev libavformat-dev libboost-all-dev libboost-date-time-dev libbullet-dev libfaac-dev libglew-dev libgsm1-dev liblapack-dev liblog4cxx-dev libmpfr-dev libode-dev libogg-dev libpcrecpp0v5 libpcre3-dev libqhull-dev libqt4-dev libsoqt-dev-common libsoqt4-dev libswscale-dev libswscale-dev libvorbis-dev libx264-dev libxml2-dev libxvidcore-dev

# # Build OpenSceneGraph-3.4.
# WORKDIR /
# RUN apt-get install libcairo2-dev libjasper-dev libpoppler-glib-dev libsdl2-dev libtiff5-dev libxrandr-dev
# RUN git clone https://github.com/openscenegraph/OpenSceneGraph.git --branch OpenSceneGraph-3.4
# RUN mkdir /OpenSceneGraph/build
# WORKDIR /OpenSceneGraph/build
# RUN cmake .. -DDESIRED_QT_VERSION=4
# RUN make -j$(nproc)
# RUN make install

# # More OpenRAVE setup.
# RUN pip install --upgrade --user sympy==0.7.1
# RUN apt remove python-mpmath

# # Install OpenRAVE.
# RUN apt-get update ; apt-get install ros-indigo-openrave
# RUN echo "PYTHONPATH=$PYTHONPATH:`openrave-config --python-dir`" >> /root/.bashrc

