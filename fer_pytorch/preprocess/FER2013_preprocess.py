import os
import csv
import h5py
import  numpy as np
import pandas as pd
import  argparse
from  fer_pytorch.utils.common import create_dir_maybe
from  fer_pytorch.utils.split_dataset import split_to_n_pieces_with_train_val
import tqdm
import  cv2

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--anno_path',
                     type=str,
                     default='../data/FER2013/fer2013.csv',
                     help='annotation file path')
    arg.add_argument('--save_dir', type=str, default='../data/FER2013/')
    return arg.parse_args()

def _row_to_sample(row):
    temp_list = []
    for pixel in row[1].split():
        temp_list.append(int(pixel))
    I = np.asarray(temp_list)
    I = I.reshape(48, 48)
    #I = I[:, :, np.newaxis]
    #img = cv2.merge((I, I, I))
    #I = img[:, :, 0].reshape(-1)
    I = I.reshape(-1)
    return I.tolist(), int(row[0])

def parse_fer2013_csv(file):
    '''
    We use 'Training & PublicTest' for  train, 'PublicTest' for val
    '''
    training_x = []
    training_y = []
    val_x = []
    val_y = []
    with open(file, 'r') as csvin:
        data = csv.reader(csvin)
        for row in tqdm.tqdm(data):
            if row[-1] in ['Training', "PublicTest"]:
                x, y = _row_to_sample(row)
                training_x.append(x)
                training_y.append(y)
            if row[-1] == 'PrivateTest':
                x, y = _row_to_sample(row)
                val_x.append(x)
                val_y.append(y)
    return  training_x,training_y,val_x,val_y


def convert_to_h5_dataset(anno_path, h5_save_dir):
    print('====> parse data..')
    training_x,training_y,val_x,val_y =parse_fer2013_csv(anno_path)
    h5_save_path = h5_save_dir+'/fer2013_train.h5'
    print('====>train save to: {}'.format(h5_save_path))
    train_data = h5py.File(h5_save_path, 'w')
    train_data.create_dataset("pixel", dtype = 'uint8', data = training_x)
    train_data.create_dataset("label", dtype = 'int64', data = training_y)
    train_data.close()

    h5_save_path = h5_save_dir + '/fer2013_val.h5'
    print('====>val save to: {}'.format(h5_save_path))
    val_data = h5py.File(h5_save_path, 'w')
    val_data.create_dataset("pixel", dtype = 'uint8', data = val_x)
    val_data.create_dataset("label", dtype = 'int64', data = val_y)
    val_data.close()

    print('====> finished!')


if __name__ =='__main__':
    args = get_args()
    convert_to_h5_dataset(args.anno_path, args.save_dir)