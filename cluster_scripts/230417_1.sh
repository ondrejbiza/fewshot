nvidia-smi

export RNDF_SOURCE_DIR=/mnt/yanjing-nfs-hdd/ondrej/relational_ndf/src/rndf_robot
export MESA_GL_VERSION_OVERRIDE=3.3
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

apt update
apt install -y bash nano git cmake wget htop software-properties-common
apt install -y libffi-dev
apt install -y ffmpeg libsm6 libxext6

# apt install -y libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6
# apt install -y freeglut3-dev libosmesa6-dev
# apt install -y libegl1 libegl-dev
# apt install -y libglfw3-dev libgles2-mesa-dev
pip install PyOpenGL

cd /mnt/yanjing-nfs-hdd/ondrej/relational_ndf
pip install numpy cython
pip install -e .

cd /mnt/yanjing-nfs-hdd/ondrej/fewshot
./shell_scripts/run_rndf_1_demo.sh
