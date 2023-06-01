from typing import Tuple


DESK_CENTER: Tuple[float, float, float] = (-0.527, -0.005, -0.075)
WORKSPACE_SIZE: float = 0.4
HEIGHT_EPS: float = 0.005

NDF_BRANCH_POSITION = (-0.034, 0.094, 0.2167)
NDF_MUG_POSITION = (0.064, 0.104, 0.101)
NDF_BOX_POSITION = (0.080, 0.120, 0.009)

NDF_MUGS_PCA_PATH = "data/pcas/230315_ndf_mugs_scale_pca_8_dim_alp_0_01.pkl"
NDF_TREES_PCA_PATH = "data/pcas/230315_ndf_trees_scale_pca_8_dim_alp_0_01.pkl"
NDF_BOWLS_PCA_PATH = "data/pcas/230315_ndf_bowls_scale_pca_8_dim_alp_0_01.pkl"
NDF_BOTTLES_PCA_PATH = "data/pcas/230328_ndf_bottles_scale_pca_8_dim_alp_0_01.pkl"
SIMPLE_TREES_PCA_PATH = "data/pcas/230315_simple_trees_scale_pca_8_dim_alp_0_01.pkl"
NDF_BRUSH_PCA_PATH = "data/data_new/pca_brush_ndf.pkl"
NDF_BRUSH_PCA_PATH_2 = "data/pca_brush_new.pkl"
NDF_BRUSH_PCA_PATH_3 = "data/brush_2.pkl"
BOXES_PCA_PATH = "data/pcas/230315_boxes_scale_pca_8_dim_alp_0_01.pkl"
NDF_CUBE_PCA_PATH = "data/pcas/230315_boxes_scale_pca_8_dim_alp_0_01.pkl"

NDF_MUGS_INIT_SCALE = 0.7
NDF_BOWLS_INIT_SCALE = 0.8
NDF_BOTTLES_INIT_SCALE = 1.
NDF_TREES_INIT_SCALE = 1.
BOXES_INIT_SCALE = 1.
SIMPLE_TREES_INIT_SCALE = 1.

NDF_BRUSHES_INIT_SCALE = 0.8
NDF_CUBE_INIT_SCALE = 1

# BRUSHES_INIT_SCALE = 1
TASKS_DESCRIPTION = "[mug_tree, bowl_on_mug, bottle_in_box, brushes, brush_in_boel, brush_on_cube]"
