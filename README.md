COCO:

python demo.py --config-file ../configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl

python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl

ADE20k

python demo.py --config-file ../configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/panoptic/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_e0c58e.pkl

python demo.py --config-file ../configs/ade20k/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/instance/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_92dae9.pkl

python demo.py --config-file ../configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --input ~/Pictures/test1.png ~/Pictures/test2.png ~/Pictures/test3.png ~/Pictures/test4.png ~/Pictures/test5.png --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_large_IN21k_384_bs16_160k_res640/model_final_6b4a3a.pkl
