import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
import shutil
from random import shuffle

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    ##### ファイルディレクトリ名取得
    ##### waymoフォルダ
    basefolder_dir = data_dir
    extension = 'tfrecord'
    basefile_dir = os.path.join(basefolder_dir, 'waymo', 'training_and_validation' , '*.' + extension)
    print('basefile_dir', basefile_dir)

    ##### 分割先ディレクトリ名設定
    train_dir = os.path.join(basefolder_dir, 'train')
    val_dir   = os.path.join(basefolder_dir, 'val')
    test_dir  = os.path.join(basefolder_dir, 'test')
    print('train_dir', train_dir)
    
    ##### 分割先ディレクトリ作成    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    ##### ファイル配分割合 ['train', 'val', 'test']
    dir_ratio = [0.7, 0.2, 0.1]
        
    ##### ファイル群取得　glob.globを使う。
    files = glob.glob(os.path.join(basefile_dir))
    print('files', files)
    
    ##### シャッフル
    shuffle(files)
    
    ##### ファイル数算出
    total_files = len(files)
    num_train = int(dir_ratio[0]/sum(dir_ratio) * total_files)
    num_val = int(dir_ratio[1]/sum(dir_ratio) * total_files)
    print('total_files', total_files)
    print('num_train', num_train)
    print('num_val', num_val)
    
    ##### ファイル移動
    for file in files[:num_train]:
        new_path = shutil.move(file, train_dir)
    for file in files[num_train:num_train+num_val]:
        new_path = shutil.move(file, val_dir)
    for file in files[num_train+num_val:]:
        new_path = shutil.move(file, test_dir)
    
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)