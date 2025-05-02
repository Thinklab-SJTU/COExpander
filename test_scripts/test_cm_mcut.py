import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv, COExpanderMCutSolver, COExpanderDecoder
)

# solving settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 1 # Ds
SAMPLING_NUM = 1 # S
FINETUNE = False # FT

# local search settings
USE_RLSA = False
RLSA_SETTINGS_DICT = {
    "BA-SMALL": (0.01, 20, 1000, 1000),
    "BA-LARGE": (1, 100, 1000, 1000),
    "BA-GIANT": (1, 200, 1000, 1000)
}

# test files and pretrained files
TEST_TYPE = "BA-SMALL"
TEST_FILE_DICT = {
    "BA-SMALL": "test_dataset/mcut/mcut_ba-small_gurobi-60s_727.844.txt",
    "BA-LARGE": "test_dataset/mcut/mcut_ba-large_gurobi-300s_2936.886.txt",
    "BA-GIANT": "test_dataset/mcut/mcut_ba-giant_gurobi-3600s_7217.900.txt"
}
WEIGHT_PATH_DICT = {
    "BA-SMALL": {
        False: "weights/cm_mcut_ba-small_sparse.pt",
        True: "weights/cm_mcut_ba-small_sparse_finetune.pt",
    },
    "BA-LARGE": {
        False: "weights/cm_mcut_ba-large_sparse.pt",
        True: "weights/cm_mcut_ba-large_sparse_finetune.pt",
    },
    "BA-GIANT": {
        False: "weights/cm_mcut_ba-large_sparse.pt",
        True: "weights/cm_mcut_ba-large_sparse_finetune.pt",
    }
}

# main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    model = COExpanderCMModel(
        env=COExpanderEnv(
            task="MCut", mode="solve", sparse_factor=1, device="cuda",
        ),
        encoder=GNNEncoder(
            task="MCut",
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
                "rlsa_t": rlsa_settings[3]
            }
        ),
        inference_steps=INFERENCE_STEP,
        determinate_steps=DETERMINATE_STEP,
        weight_path=WEIGHT_PATH_DICT[TEST_TYPE][FINETUNE]
    )
    solver = COExpanderMCutSolver(model=model)
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], show_time=True, ref=True)
    solver.solve(show_time=True, sampling_num=SAMPLING_NUM)
    print(solver.evaluate(calculate_gap=True))