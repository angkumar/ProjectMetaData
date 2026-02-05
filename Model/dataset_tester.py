import h5py

file_path = "/Users/itsak/Downloads/histopathologic-cancer-detection/camelyonpatch_level_2_split_train_x.h5"

with h5py.File(file_path, "r") as f:
    print("Keys:", list(f.keys()))
    
    x = f["x"]
    print("Shape:", x.shape)
    print("Dtype:", x.dtype)