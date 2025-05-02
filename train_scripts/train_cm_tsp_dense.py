import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit import Trainer
from co_expander import (
    COExpanderCMModel, TSPGNNEncoder, COExpanderEnv, COExpanderDecoder
)


if __name__ == "__main__":
    model = COExpanderCMModel(
        env=COExpanderEnv(
            task="TSP",
            mode="train",
            train_data_size=64000,
            val_data_size=1280,
            train_batch_size=16,
            val_batch_size=1,
            num_workers=4,
            sparse_factor=-1,
            device="cuda",
            train_folder="path/to/train/folder",
            val_path="path/to/val/file",
            store_data=True # set False if your server's memory is not large enough.
        ),
        encoder=TSPGNNEncoder(
            sparse=True,
            num_layers=12,
            hidden_dim=256
        ),
        decoder=COExpanderDecoder(),
        learning_rate=0.0002,
        # weight_path="path/to/pretrained/file"
    )

    trainer = Trainer(model=model, devices=[0])
    trainer.model_train()