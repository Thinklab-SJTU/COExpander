import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv, COExpanderMISSolver, COExpanderDecoder
)

# solving settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 1 # Ds
SAMPLING_NUM = 1 # S
BEAM_SIZE = -1 # Beam Search

# local search settings
USE_RLSA = False
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 5, 1000, 500, 1.02, 0.3),
    "RB-LARGE": (0.01, 5, 1000, 1000, 1.02, 0.3),
    "RB-GIANT": (0.01, 5, 1000, 1000, 1.02, 0.3),
    "ER-700-800": (0.2, 10, 1000, 1000, 1.001, 0.3),
    "ER-1400-1600": (0.2, 10, 1000, 1000, 1.001, 0.3),
    "SATLIB": (0.01, 5, 1000, 1000, 1.02, 0.3),
}

# test files and pretrained files
TEST_TYPE = "RB-SMALL"
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mis/mis_rb-small_kamis-60s_20.090.txt",
    "RB-LARGE": "test_dataset/mis/mis_rb-large_kamis-60s_43.004.txt",
    "RB-GIANT": "test_dataset/mis/mis_rb-giant_kamis-60s_49.260.txt",
    "ER-700-800": "test_dataset/mis/mis_er-700-800_kamis-60s_44.969.txt",
    "ER-1400-1600": "test_dataset/mis/mis_er-1400-1600_kamis-60s_50.938.txt",
    "SATLIB": "test_dataset/mis/mis_satlib_kamis-60s_425.954.txt"
}
WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/cm_mis_rb-small_sparse.pt",
    "RB-LARGE": "weights/cm_mis_rb-large_sparse.pt",
    "RB-GIANT": "weights/cm_mis_rb-large_sparse.pt",
    "ER-700-800": "weights/cm_mis_er-700-800_sparse.pt",
    "ER-1400-1600": "weights/cm_mis_er-700-800_sparse.pt",
    "SATLIB": "weights/cm_mis_satlib_sparse.pt"
}

# main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    model = COExpanderCMModel(
        env=COExpanderEnv(
            task="MIS", mode="solve", sparse_factor=1, device="cuda",
        ),
        encoder=GNNEncoder(
            task="MIS",
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
        beam_size=BEAM_SIZE,
        weight_path=WEIGHT_PATH_DICT[TEST_TYPE]
    )
    solver = COExpanderMISSolver(model=model)
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], show_time=True, ref=True)
    solver.solve(show_time=True, sampling_num=SAMPLING_NUM)
    print(solver.evaluate(calculate_gap=True))