import h5py
import os

def explore_h5_file(filename):
    with h5py.File(filename, 'r') as f:
        def explore_group(group, prefix=""):
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"{prefix}/{key} (Dataset) Shape: {item.shape} Dtype: {item.dtype}")
                elif isinstance(item, h5py.Group):
                    print(f"{prefix}/{key} (Group)")
                    explore_group(item, prefix=f"{prefix}/{key}")
                    
        explore_group(f)


path = os.getcwd()

milano = path + '/dataset/' + 'milano.h5'
trento = path + '/dataset/' + 'trento.h5'
print("[Milano Dataset]")
explore_h5_file(milano)

print("\n[Trento Dataset]")
explore_h5_file(trento)