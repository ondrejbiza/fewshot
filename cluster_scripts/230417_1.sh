# run using det command run --config-file 230417_1.yaml "$(< 230417_1.sh)"
nvidia-smi

export RNDF_SOURCE_DIR=/mnt/yanjing-nfs-hdd/ondrej/relational_ndf/src/rndf_robot
export MESA_GL_VERSION_OVERRIDE=3.3
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

cd /mnt/yanjing-nfs-hdd/ondrej/fewshot
bash shell_scripts/run_rndf_1_demo.sh
