import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from co_expander import (
    COExpanderCMModel, TSPGNNEncoder, COExpanderEnv, COExpanderTSPSolver, COExpanderDecoder
)

# problem settings
NODES_NUM = 100
SPARSE_FACTOR = -1

# solving settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 1 # Ds (must be 1 or 3)
SAMPLING_NUM = 1 # S

# local search settings
USE_2OPT = False

# test files & pretrained files
TEST_FILE_DICT = {
    50: "test_dataset/tsp/tsp50_concorde_5.688.txt",
    100: "test_dataset/tsp/tsp100_concorde_7.756.txt",
    500: "test_dataset/tsp/tsp500_concorde_16.546.txt",
    1000: "test_dataset/tsp/tsp1000_concorde_23.118.txt",
    10000: "test_dataset/tsp/tsp10000_lkh_500_71.755.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/cm_tsp50_dense.pt",
    100: "weights/cm_tsp100_dense.pt",
    500: "weights/cm_tsp500_sparse.pt",
    1000: "weights/cm_tsp1k_sparse.pt",
    10000: "weights/cm_tsp10k_sparse.pt"
}

# main
if __name__ == "__main__":
    solver = COExpanderTSPSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="TSP", sparse_factor=SPARSE_FACTOR, device="cuda"
            ),
            encoder=TSPGNNEncoder(sparse=SPARSE_FACTOR>0),
            decoder=COExpanderDecoder(
                decode_kwargs={"use_2opt": USE_2OPT}
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))