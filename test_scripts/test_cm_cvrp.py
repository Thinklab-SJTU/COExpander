import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv, COExpanderCVRPSolver, COExpanderDecoder
)

# solving settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 1 # Ds
SAMPLING_NUM = 1 # S

# local search settings
USE_LS = True

# test files and pretrained files
TEST_TYPE = "CVRP-50"
TEST_FILE_DICT = {
    "CVRP-50": "test_dataset/cvrp/cvrp50_hgs-1s_10.366.txt",
    "CVRP-100": "test_dataset/cvrp/cvrp100_hgs-20s_15.563.txt",
    "CVRP-200": "test_dataset/cvrp/cvrp200_hgs-60s_19.630.txt",
    "CVRP-500": "test_dataset/cvrp/cvrp500_hgs-300s_37.154.txt",
}
WEIGHT_PATH_DICT = {
    "CVRP-50": "weights/cm_cvrp50_dense.pt",
    "CVRP-100": "weights/cm_cvrp100_dense.pt",
    "CVRP-200": "weights/cm_cvrp200_dense.pt",
    "CVRP-500": "weights/cm_cvrp500_dense.pt",
}

# main
if __name__ == "__main__":
    model = COExpanderCMModel(
        env=COExpanderEnv(
            task="CVRP", mode="solve", sparse_factor=-1, device="cuda",
        ),
        encoder=GNNEncoder(
            task="CVRP",
            sparse=False,
            block_layers=[2, 4, 4, 2],
            hidden_dim=256
        ),
        decoder=COExpanderDecoder(
            decode_kwargs={"use_ls": USE_LS}
        ),
        inference_steps=INFERENCE_STEP,
        determinate_steps=DETERMINATE_STEP,
        weight_path=WEIGHT_PATH_DICT[TEST_TYPE]
    )
    solver = COExpanderCVRPSolver(model=model)
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], show_time=True, ref=True)
    solver.solve(show_time=True, sampling_num=SAMPLING_NUM)
    print(solver.evaluate(calculate_gap=True))