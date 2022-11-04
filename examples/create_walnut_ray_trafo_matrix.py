from bayes_dip.data.walnut_utils import save_single_slice_ray_trafo_matrix

ANGULAR_SUB_SAMPLING = 20  # 1200 -> 60
PROJ_COL_SUB_SAMPLING = 6  # 768 -> 128

WALNUT_ID = 1
ORBIT_ID = 2

DATA_PATH = '../experiments/walnuts/'
OUTPUT_PATH = DATA_PATH

save_single_slice_ray_trafo_matrix(
        output_path=OUTPUT_PATH, data_path=DATA_PATH,
        walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
        angular_sub_sampling=ANGULAR_SUB_SAMPLING,
        proj_col_sub_sampling=PROJ_COL_SUB_SAMPLING)
