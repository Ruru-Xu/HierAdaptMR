import os
import glob
from scipy.io import loadmat
import h5py
import scipy
root = '/media/ruru/ad31566c-e032-4ffa-a8cf-751b9dbab424/work/CMRxRecon2025/preprocess1/train'

def loadmat(filename):
    """
    Load .mat file using appropriate method based on file format.
    Tries scipy.io.loadmat first (for v7.2 and earlier),
    falls back to h5py for v7.3 format files.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")

    try:
        # First try scipy.io.loadmat for older format files
        data = scipy.io.loadmat(filename)
        # Remove scipy metadata keys that start with '__'
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        # print(f"Loaded {filename} using scipy.io.loadmat")
        return data

    except (NotImplementedError, ValueError) as e:
        # If scipy fails, try h5py for v7.3 format
        try:
            # print(f"scipy.io failed, trying h5py for {filename}")
            with h5py.File(filename, 'r') as f:
                data = {}
                for k, v in f.items():
                    if isinstance(v, h5py.Dataset):
                        data[k] = v[()]
                    elif isinstance(v, h5py.Group):
                        data[k] = loadmat_group(v)
            # print(f"Loaded {filename} using h5py")
            return data

        except Exception as h5_error:
            raise RuntimeError(f"Could not load {filename} with either method. "
                               f"scipy error: {e}, h5py error: {h5_error}")

def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data


f = []
# fileName = 'T2w/TrainingSet/FullSample'
# files = sorted(glob.glob(os.path.join(root, fileName, "**/*T2w*.mat"), recursive=True))
files = sorted(glob.glob(os.path.join(root, "**/**.h5"), recursive=True))
for f in files:
    hf = loadmat(f)
    kspace_shape = hf["kspace"].shape
    f_name = f.split('/')[-1]
    print(f'{f_name}    {kspace_shape}')

