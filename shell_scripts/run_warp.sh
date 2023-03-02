CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class mug --child_class bowl \
    --exp bowl_on_mug_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bowl_on_mug_relation --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright &> outputs/bowl_on_mug_upright.txt

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class mug --child_class bowl \
    --exp bowl_on_mug_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bowl_on_mug_relation --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/bowl_on_mug_any.txt

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright &> outputs/mug_on_tree_upright.txt

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/mug_on_tree_any.txt

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class syn_container --child_class bottle \
    --exp bottle_in_container_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_container.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bottle.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bottle_in_container_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --pc_reference child \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright &> outputs/bottle_in_box_upright.txt

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class syn_container --child_class bottle \
    --exp bottle_in_container_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_container.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bottle.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bottle_in_container_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --pc_reference child \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/bottle_in_box_any.txt
