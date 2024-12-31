

Setting up the environment: 
	conda env create -f environment.yml

I'm using the SAM version here: https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth



PCD Segmentation:
  For segmenting and concatenating pointclouds from multiple camera views. 
  	- Swap in the filename for the pcd scene
	- Uncomment the config for the correct experiment in cell 4 and run cells in order
	- Select points on the visualization on cell 6 by picking a camera and part with the radio buttons. 
	  Re-selecting a camera resets all parts' points for that camera 
	- Produces image masks and segmented pointclouds per camera, saves them per object part
  Code for picking and visualizing a grasp is at the bottom of this notebook. Just run those cells in order. 
	
Parts-Based Warp Reconstruction from Segmented PCD: 
	- Swap in the saved output from PCD Segmentation in cell 2, select experiment in cell 3, and run cells in order
	
Parts-Based Interaction Points
	- Swap in the segmented pcds and output of warp reconstruction in cell 2, select experiment in cell 3, run cells in order 

Parts-Based Infer Final Pose: 
	- Run PCD Segmentation and warp reconstruction on your test scene. Load those files and your demo's interaction point file in cell 2
	- Set experiment in cell 3
	- Run cells in order

Whole Object PCD Segmentation: 
	- Swap in the filename for the pcd scene
	- Uncomment the config for the correct experiment in cell 4 and run cells in order
	- Select points on the visualization on cell 6 by picking a camera and part with the radio buttons. 
	  Re-selecting a camera resets all parts' points for that camera 
	- Produces image masks and segmented pointclouds per camera, saves them per object part
  Code for picking and visualizing a grasp is at the bottom of this notebook. Just run those cells in order. 
Get Whole Object Reconstruction: 
	- Swap in the saved output from PCD Segmentation in cell 2, select experiment in cell 3, and run cells in order
Get Whole Object Interaction Points 
 	- Swap in the segmented pcds and output of warp reconstruction in cell 2, select experiment in cell 3, run cells in order 
Get Whole Object Inferred Final Pose:
	- Run PCD Segmentation and warp reconstruction on your test scene. Load those files and your demo's interaction point file in cell 2
	- Set experiment in cell 3
	- Run cells in order 

Dino-V2 Clicked Point Transfer (doesn't work well): 
 	- Load demo and test images
 	- Uncomment the config for the correct experiment in cell 4 and run cells in order
	- Select points on the visualization on cell 6 by picking a camera and part with the radio buttons. 
	
