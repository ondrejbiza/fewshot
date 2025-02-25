
#!/bin/bash


#SEED=0
echo "Running mug bowl experiments"
eval "$(conda shell.bash hook)"
conda activate part_based
cd ~/relational_ndf/
source rndf_env.sh
cd ~/fewshot
export PATH=$PATH:~/v-hacd/app/build/


SOURCE_FILE='20240426-000022_10'
TARGET_FILE='20240501-191613_10'
SEED=2024

foo=`seq 0 1 10`
for i in $foo
do
    python -m scripts.compare_part_whole --parent_class mug --child_class bowl \
        --exp bowl_on_mug_upright_pose_new \
        --canon_source_file_stamp $SOURCE_FILE\
        --canon_target_file_stamp $TARGET_FILE\
        --is_parent_shapenet_obj --is_child_shapenet_obj \
        --num_iterations 50 \
        --rel_demo_exp release_demos/bowl_on_mug_relation --pybullet_server \
        --opt_iterations 650 \
        --demo_idx $i \
        --seed $SEED \
        --parent_load_pose_type random_upright --child_load_pose_type any_pose &> "outputs/part_bowl_on_mug_any_1_demo${i}.txt"

    python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
        --exp bowl_on_mug_upright_pose_new \
        --parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
        --child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
        --is_parent_shapenet_obj --is_child_shapenet_obj \
        --rel_demo_exp release_demos/bowl_on_mug_relation \
        --pybullet_server \
        --opt_iterations 650 \
        --num_iterations 50 \
        --seed $SEED \
        --n_demos 1 \
        --demo_idx $i \
        --new_descriptors \
        --parent_load_pose_type random_upright --child_load_pose_type any_pose &> "outputs/rndf_bowl_on_mug_any_1_demo${i}.txt"
done

echo "Complete"
