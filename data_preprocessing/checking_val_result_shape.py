import os
import glob
from scipy.io import loadmat
import h5py
my_root = '/media/ruru/ad31566c-e032-4ffa-a8cf-751b9dbab424/work/CMRxRecon2025/ValSet/testing/output/Submission_testing1/TaskR1/MultiCoil'
# official_root = '/media/ruru/ad31566c-e032-4ffa-a8cf-751b9dbab424/work/CMRxRecon2025/ValSet/testing/Submission_GRAPPA_R1/TaskR1/MultiCoil'

f = []
for modility in ["BlackBlood", "Cine", "Flow2d", "LGE", "Mapping", "Perfusion", "T1rho", "T1w", "T2w"]:
    f += sorted([file for file in glob.glob(os.path.join(my_root, modility, 'ValidationSet', 'UnderSample_TaskR1', '**/*.mat'), recursive=True)])
    print('Input data:{}\n Total files: {}'.format(my_root, len(f)))

    for my_file in f:
        official_file = my_file.replace('output/Submission_testing1', 'Submission_GRAPPA_R1')
        original_file = my_file.replace('ValSet/testing/output/Submission_testing1', 'ValSet')
        my_kspace = loadmat(my_file)['img4ranking']
        offical_kspace = loadmat(official_file)['img4ranking']
        origianl_kspace = h5py.File(original_file)['kus']
        if my_kspace.shape != offical_kspace.shape:
            print(my_file)
            print(f'my_kspace:{my_kspace.shape}, official_kspace:{offical_kspace.shape}, original_kspace:{origianl_kspace.shape}')

