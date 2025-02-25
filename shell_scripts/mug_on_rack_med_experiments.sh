
#!/bin/bash

echo "Running med tree experiments"
eval "$(conda shell.bash hook)"
conda activate part_based
cd ~/relational_ndf/
source rndf_env.sh
cd ~/fewshot
export PATH=$PATH:~/v-hacd/app/build/

SOURCE_FILE='20240501-191613_10'
TARGET_FILE='20240503-041049_10'


foo=`seq 0 1 1`
for i in $foo 
do
    python -m scripts.compare_part_whole --parent_class syn_rack_med_1 --child_class mug \
        --exp mug_on_rack_med_1_pose_new \
        --canon_source_file_stamp $SOURCE_FILE\
        --canon_target_file_stamp $TARGET_FILE\
        --is_child_shapenet_obj \
        --num_iterations 20 \
        --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
        --opt_iterations 650 \
        --demo_idx $i \
        --parent_load_pose_type random_upright --child_load_pose_type any_pose #&> "outputs/_mug_on_tree_upright_1_demo${i}.txt"

    python -m scripts.run_rndf_no_viz --parent_class syn_rack_med_1 --child_class mug \
        --exp mug_on_rack_upright_pose_new \
        --parent_model_path ndf_vnn/rndf_weights/ndf_med_rack_100.pth \
        --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
        --is_child_shapenet_obj \
        --rel_demo_exp release_demos/mug_on_rack_relation \
        --pybullet_server \
        --opt_iterations 650 \
        --num_iterations 20 \
        --n_demos 1 \
        --demo_idx $i \
        --new_descriptors \
        --parent_load_pose_type random_upright --child_load_pose_type any_pose #&> "outputs/rndf_mug_on_tree_upright_1_demo${i}.txt
done