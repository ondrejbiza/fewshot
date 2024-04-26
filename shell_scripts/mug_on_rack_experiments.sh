python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 0 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo0.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 0 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo0.txt

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 1 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo1.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 1 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo1.txt

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 2 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo2.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 2 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo2.txt

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 4 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo4.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 4 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo4.txt

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 5 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo5.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 5 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo5.txt

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 6 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo6.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 6 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo6.txt

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 7 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo7.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 7 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo7.txt

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 8 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo8.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 8 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo8.txt

python -m scripts.compare_part_whole --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --canon_source_file_stamp 20240425-225313_6\
    --canon_target_file_stamp 20240320-032402\
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --num_iterations 20 \
    --rel_demo_exp release_demos/mug_on_rack_relation  --pybullet_server \
    --opt_iterations 650 \
    --demo_idx 9 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo9.txt

python -m scripts.run_rndf_no_viz --parent_class mug --child_class bowl \
	--exp mug_on_rack_upright_pose_new \
	--parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
	--is_child_shapenet_obj \
	--rel_demo_exp release_demos/mug_on_rack_relation \
	--pybullet_server \
	--opt_iterations 650 \
	--num_iterations 20 \
	--n_demos 1 \
	--demo_idx 9 \
	--new_descriptors \
	--parent_load_pose_type random_upright --child_load_pose_type any_pose &> outputs/rndf_mug_on_tree_upright_1_demo9.txt