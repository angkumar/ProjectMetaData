import h5py
import numpy as np
with h5py.File("Training_Data/pcam/training_split.h5", "r") as f:
    print("Images shape:", f["x"].shape)

with h5py.File("Training_Data/Labels/camelyonpatch_level_2_split_train_y.h5", "r") as f:
    print("Labels shape:", f["y"].shape)
    print("Unique labels:", np.unique(f["y"][:100]))  # check first 100
