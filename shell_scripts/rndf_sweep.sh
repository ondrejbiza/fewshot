#!/bin/bash

echo "Running rndf sweep experiments"
eval "$(conda shell.bash hook)"
conda activate part_based
cd ~/relational_ndf/
source rndf_env.sh
cd ~/fewshot
export PATH=$PATH:~/v-hacd/app/build/

SOURCE_FILE='20240501-191613_10'
TARGET_FILE='20240430-043520_10'
SEED=2024

python -m scripts.run_rndf_no_viz --parent_class syn_rack_easy --child_class mug \
        --exp mug_on_rack_sweep \
        --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
        --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
        --is_child_shapenet_obj \
        --rel_demo_exp release_demos/mug_on_rack_relation \
        --pybullet_server \
        --opt_iterations 650 \
        --num_iterations 20 \
        --seed $SEED \
        --n_demos 1 \
        --demo_idx 0 \
        --sweep \
        --new_descriptors \
        --parent_load_pose_type random_upright --child_load_pose_type any_pose #&> "outputs/rndf_mug_on_tree_upright_1_demo${i}.txt"
