nvidia-smi

export RNDF_SOURCE_DIR=/mnt/yanjing-nfs-hdd/ondrej/relational_ndf/src/rndf_robot
export MESA_GL_VERSION_OVERRIDE=3.3
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

cd /mnt/yanjing-nfs-hdd/ondrej/fewshot
echo "test" > /tmp/test.txt
cp /tmp/test.txt outputs/
# ./shell_scripts/run_rndf_1_demo.sh
