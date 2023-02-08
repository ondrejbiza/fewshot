FROM library/ubuntu:14.04

# Utilities.
RUN apt update ; apt install -y nano git cmake wget htop software-properties-common curl build-essential checkinstall dpkg

# 'tzdata' have an annoying prompt.
RUN apt update ; DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt install -y tzdata

# Compile Python 2.7.18. We can't use apt install because of https://github.com/pypi/support/issues/978.
RUN apt update ; apt install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
RUN wget https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
RUN tar xzf Python-2.7.18.tgz
WORKDIR /Python-2.7.18
RUN ./configure --enable-optimizations
RUN make install
WORKDIR /

# Install pip.
RUN apt update ; apt install -y libffi-dev libssl-dev
RUN curl -fSL "https://bootstrap.pypa.io/pip/2.7/get-pip.py" | python

# Install ROS Indigo.
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt update ; apt install -y ros-indigo-desktop-full
RUN echo "source /opt/ros/indigo/setup.bash" >> /root/.bashrc
RUN rosdep init
RUN rosdep update
RUN apt update ; apt install -y python-rosinstall
RUN apt update ; apt install -y ros-indigo-moveit-kinematics

# Install OpenRAVE prerequisites.
# Missing: minizip, liblog4cxx-dev, libpcrecpp0v5. Probably doesn't matter...
RUN apt update ; apt install -y cmake g++ git ipython python-dev python-h5py python-numpy python-scipy qt4-dev-tools
RUN apt update ; apt install -y libassimp-dev libavcodec-dev libavformat-dev libavformat-dev libboost-all-dev libboost-date-time-dev libbullet-dev libfaac-dev libglew-dev libgsm1-dev liblapack-dev libmpfr-dev libode-dev libogg-dev libpcre3-dev libqhull-dev libqt4-dev libsoqt-dev-common libsoqt4-dev libswscale-dev libswscale-dev libvorbis-dev libx264-dev libxml2-dev libxvidcore-dev

# Build OpenSceneGraph-3.4.
WORKDIR /
RUN apt update ; apt install -y libsdl2-2.0-0
RUN apt update ; apt install -y libsdl2-dev
RUN apt update ; apt install -y libcairo2-dev libjasper-dev libpoppler-glib-dev libtiff5-dev libxrandr-dev
RUN git clone https://github.com/openscenegraph/OpenSceneGraph.git --branch OpenSceneGraph-3.4
RUN mkdir /OpenSceneGraph/build
WORKDIR /OpenSceneGraph/build
RUN cmake .. -DDESIRED_QT_VERSION=4
RUN make -j$(nproc)
RUN make install
WORKDIR /

# More OpenRAVE setup.
RUN pip install --upgrade sympy==0.7.1
RUN apt remove python-mpmath

# Install OpenRAVE.
RUN apt update ; apt install -y ros-indigo-openrave
# RUN openrave-config --python-dir
# RUN echo "PYTHONPATH=$PYTHONPATH:`openrave-config --python-dir`" >> /root/.bashrc
