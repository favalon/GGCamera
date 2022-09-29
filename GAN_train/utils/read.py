import h5py
from torch import from_numpy


def read_h5py_dataset(data_path, keys, max_n=0):
    f = h5py.File(data_path, 'r')
    read = []
    for key in keys:
        n = f[key].shape[0] if max_n <= 0 else min(max_n, f[key].shape[0])
        read.append(from_numpy(f[key][:n]))

    f.close()

    return tuple(read)
