GPU=1

CUDA_VISIBLE_DEVICES=$GPU \
    python -m scripts.run_warp --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --ablate-no-warp \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/warp_1_demo_mug_on_tree_any_ablate_no_warp.txt

CUDA_VISIBLE_DEVICES=$GPU \
    python -m scripts.run_warp --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --ablate-no-scale \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/warp_1_demo_mug_on_tree_any_ablate_no_scale.txt

CUDA_VISIBLE_DEVICES=$GPU \
    python -m scripts.run_warp --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --ablate-no-pose-training \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/warp_1_demo_mug_on_tree_any_ablate_no_pose_training.txt

CUDA_VISIBLE_DEVICES=$GPU \
    python -m scripts.run_warp --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --ablate-no-size-reg \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/warp_1_demo_mug_on_tree_any_ablate_no_size_reg.txt
