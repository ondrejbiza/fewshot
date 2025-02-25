#!/bin/bash

echo "Running mug sweep experiments"
eval "$(conda shell.bash hook)"
conda activate part_based
cd ~/relational_ndf/
source rndf_env.sh
cd ~/fewshot
export PATH=$PATH:~/v-hacd/app/build/

SOURCE_FILE='20240501-191613_10'
TARGET_FILE='20240430-043520_10'
SEED=2024

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_sweep \
    --canon_source_file_stamp $SOURCE_FILE\
    --canon_target_file_stamp $TARGET_FILE\
    --is_child_shapenet_obj \
    --num_iterations 20 \
    --seed $SEED \
    --sweep true\
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 0 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose #&> "outputs/_mug_on_tree_upright_1_demo${i}.txt"

# python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
#     --exp mug_on_rack_sweep \
#     --canon_source_file_stamp $SOURCE_FILE\
#     --canon_target_file_stamp $TARGET_FILE\
#     --is_child_shapenet_obj \
#     --num_iterations 20 \
#     --seed $SEED \
#     --sweep true\
#     --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
#     --opt_iterations 650 \
#     --demo_idx 7 \
# 	--parent_load_pose_type random_upright --child_load_pose_type any_pose #&> "outputs/_mug_on_tree_upright_1_demo${i}.txt"
