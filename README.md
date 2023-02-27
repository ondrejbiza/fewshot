## Bowl on mug:

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class mug --child_class bowl \
    --exp bowl_on_mug_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bowl_on_mug_relation --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class mug --child_class bowl \
    --exp bowl_on_mug_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
    --is_parent_shapenet_obj --is_child_shapenet_obj \
    --rel_demo_exp release_demos/bowl_on_mug_relation --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose

## Mug on rack:

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type random_upright

CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.run_warp --parent_class syn_rack_easy --child_class mug \
    --exp mug_on_rack_upright_pose_new \
    --parent_model_path ndf_vnn/rndf_weights/ndf_rack.pth \
    --child_model_path ndf_vnn/rndf_weights/ndf_mug2.pth \
    --is_child_shapenet_obj \
    --rel_demo_exp release_demos/mug_on_rack_relation \
    --pybullet_server \
    --opt_iterations 650 \
    --parent_load_pose_type random_upright --child_load_pose_type any_pose

## Bottle in container:

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
    --parent_load_pose_type random_upright --child_load_pose_type random_upright

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
    --parent_load_pose_type random_upright --child_load_pose_type any_pose


# Old stuff

COCO:

python demo.py --config-file ../configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl

python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl

ADE20k

python demo.py --config-file ../configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/panoptic/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_e0c58e.pkl

python demo.py --config-file ../configs/ade20k/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/instance/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_92dae9.pkl

python demo.py --config-file ../configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_large_IN21k_384_bs16_160k_res640/model_final_6b4a3a.pkl
