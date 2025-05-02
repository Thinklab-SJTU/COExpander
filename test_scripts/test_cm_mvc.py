import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv, COExpanderMVCSolver, COExpanderDecoder
)

# solving settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 1 # Ds
SAMPLING_NUM = 1 # S

# local search settings
USE_RLSA = False
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "RB-LARGE": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "RB-GIANT": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "TWITTER": (0.01, 2, 1000, 200, 4.0, 0.2),
    "COLLAB": (0.01, 2, 1000, 300, 1.02, 0.2)
}

# test files and pretrained files
TEST_TYPE = "RB-SMALL"
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mvc/mvc_rb-small_gurobi-60s_205.764.txt",
    "RB-LARGE": "test_dataset/mvc/mvc_rb-large_gurobi-300s_968.228.txt",
    "RB-GIANT": "test_dataset/mvc/mvc_rb-giant_gurobi-3600s_2396.780.txt",
    "TWITTER": "test_dataset/mvc/mvc_twitter_gurobi-60s_85.251.txt",
    "COLLAB": "test_dataset/mvc/mvc_collab_gurobi-60s_65.086.txt"
}
WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/cm_mvc_rb-small_sparse.pt",
    "RB-LARGE": "weights/cm_mvc_rb-large_sparse.pt",
    "RB-GIANT": "weights/cm_mvc_rb-large_sparse.pt",
    "TWITTER": "weights/cm_mvc_rb-small_sparse.pt",
    "COLLAB": "weights/cm_mvc_rb-small_sparse.pt"
}

# main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    model = COExpanderCMModel(
        env=COExpanderEnv(
            task="MVC", mode="solve", sparse_factor=1, device="cuda",
        ),
        encoder=GNNEncoder(
            task="MVC",
            sparse=True,
            block_layers=[2, 4, 4, 2],
            hidden_dim=256
        ),
        decoder=COExpanderDecoder(
            decode_kwargs={
                "use_rlsa": USE_RLSA,
                "rlsa_tau": rlsa_settings[0],
                "rlsa_d": rlsa_settings[1],
                "rlsa_k": rlsa_settings[2],
                "rlsa_t": rlsa_settings[3],
                "rlsa_beta": rlsa_settings[4],
                "rlsa_alpha": rlsa_settings[5],
            }
        ),
        inference_steps=INFERENCE_STEP,
        determinate_steps=DETERMINATE_STEP,
        weight_path=WEIGHT_PATH_DICT[TEST_TYPE]
    )
    solver = COExpanderMVCSolver(model=model)
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], show_time=True, ref=True)
    solver.solve(show_time=True, sampling_num=SAMPLING_NUM)
    print(solver.evaluate(calculate_gap=True))