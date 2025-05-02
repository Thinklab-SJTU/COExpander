import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv, COExpanderATSPSolver, COExpanderDecoder
)

# solving settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 1 # Ds
SAMPLING_NUM = 1 # S

# local search settings
USE_2OPT = True

# test files and pretrained files
TEST_TYPE = "ATSP-50"
TEST_FILE_DICT = {
    "ATSP-50": "test_dataset/atsp/atsp50_uniform_lkh_1000_1.5545.txt",
    "ATSP-100": "test_dataset/atsp/atsp100_uniform_lkh_1000_1.5660.txt",
    "ATSP-200": "test_dataset/atsp/atsp200_uniform_lkh_1000_1.5647.txt",
    "ATSP-500": "test_dataset/atsp/atsp500_uniform_lkh_1000_1.5734.txt"
}
WEIGHT_PATH_DICT = {
    "ATSP-50": "weights/cm_atsp50_dense.pt",
    "ATSP-100": "weights/cm_atsp100_dense.pt",
    "ATSP-200": "weights/cm_atsp200_dense.pt",
    "ATSP-500": "weights/cm_atsp500_dense.pt"
}

# main
if __name__ == "__main__":
    model = COExpanderCMModel(
        env=COExpanderEnv(
            task="ATSP", mode="solve", sparse_factor=-1, device="cuda",
        ),
        encoder=GNNEncoder(
            task="ATSP",
            sparse=False,
            block_layers=[2, 4, 4, 2],
            hidden_dim=256
        ),
        decoder=COExpanderDecoder(
            decode_kwargs={"use_2opt": USE_2OPT}
        ),
        inference_steps=INFERENCE_STEP,
        determinate_steps=DETERMINATE_STEP,
        weight_path=WEIGHT_PATH_DICT[TEST_TYPE]
    )
    solver = COExpanderATSPSolver(model=model)
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], show_time=True, ref=True)
    solver.solve(show_time=True, sampling_num=SAMPLING_NUM)
    print(solver.evaluate(calculate_gap=True))