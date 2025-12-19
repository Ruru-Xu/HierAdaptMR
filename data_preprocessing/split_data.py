import os
import glob
import json
import argparse
from os.path import join
import torch
from creat_json import create_center_based_split
import numpy as np

def to_tensor(data: np.ndarray) -> torch.Tensor:
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

if __name__ == '__main__':
    # add argparse
    parser = argparse.ArgumentParser(description='Prepare H5 dataset for CMRxRecon series dataset')
    parser.add_argument('--output_h5_folder', type=str, default='/media/ruru/ad31566c-e032-4ffa-a8cf-751b9dbab424/work/CMRxRecon2025/preprocess',
                        help='path to save H5 dataset')
    args = parser.parse_args()

    save_folder = args.output_h5_folder

    print('## step 2: create .json file')
    create_center_based_split(save_folder)

    print('## step 3: split h5 dataset to train and val using symbolic links')
    # split dataset to train/ val according to provided json file
    split_json = '/home/ruru/Documents/work/CMR2025/cmr2025_R1/data_preprocessing/cmr25-cardiac.json'
    with open(split_json, 'r', encoding="utf-8") as f:
        split_dict = json.load(f)
    print('train files in json: ', len(split_dict['train']))
    print('val files in json: ', len(split_dict['val']))

    train_folder = os.path.join(save_folder, 'train')
    val_folder = os.path.join(save_folder, 'val')
    # if not os.path.exists(train_folder):
    #     os.makedirs(train_folder)
    # if not os.path.exists(val_folder):
    #     os.makedirs(val_folder)

    file_list = sorted(glob.glob(os.path.join(save_folder, '**/*', 'TrainingSet/FullSample', '**/*', '**/*.h5')))
    print('number of total files in folder h5_dataset: ', len(file_list))

    train_list = [ff.split('/')[-1] for ff in split_dict['train']]
    val_list = [ff.split('/')[-1] for ff in split_dict['val']]

    for ff in file_list:
        f_str_part = ff.split('/')
        save_name = f_str_part[-4] + '_' + f_str_part[-3] + '_' + f_str_part[-7] + '_' + f_str_part[-2] + '_' + f_str_part[-1]
        if save_name in train_list:
            os.symlink(ff, join(train_folder, save_name))
        elif save_name in val_list:
            os.symlink(ff, join(val_folder, save_name))

    print('Done!')
    print('number of files in h5 folder: ', len(file_list))
    print('number of symbolic link files in train folder: ', len(glob.glob(join(train_folder, '*.h5'))))
    print('number of symbolic link files in val folder: ', len(glob.glob(join(val_folder, '*.h5'))))