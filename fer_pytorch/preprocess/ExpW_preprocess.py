import os
import pandas as pd
import  argparse
from  fer_pytorch.utils.common import create_dir_maybe
from  fer_pytorch.utils.split_dataset import split_to_n_pieces_with_train_val
def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--origin_anno_path',
                     type=str,
                     default='/home/lhw/yangzhi/FaceExpRecog/2.data/ExpW/label/label.lst',
                     help='original expW annotation file path')
    arg.add_argument('--min_face_size', type = int, default = 80)
    arg.add_argument('--split_seed', type=int, default=666)
    arg.add_argument('--train_ratio', type=float, default=0.8)
    arg.add_argument('--save_dir', type=str, default='./cache/')
    return arg.parse_args()

def read_label(path):
    columns = ['fname','id','y0','x0','x1','y1','conf','label','face_s' ]
    label_info_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            d = line.split()
            face_s = abs(int(d[2]) - int(d[5])) * abs(int(d[3]) - int(d[4]))
            label_info_list.append(
                [d[0], int(d[1]),int(d[2]), int(d[3]), int(d[4]),
                 int(d[5]),float(d[6]), int(d[7]),face_s]
            )
    return pd.DataFrame(label_info_list, columns = columns)

def main():
    args = get_args()

    df = read_label(args.origin_anno_path)
    print('====> original samples: ', len(df))
    min_s = args.min_face_size**2
    df_keeped = df.copy(deep=True)
    df_keeped = df_keeped[df_keeped['face_s'] > min_s]
    print('====> min_size: {}'.format(args.min_face_size))
    print('====> drop some... keeped  samples: ', len(df_keeped))
    create_dir_maybe(args.save_dir)
    pieces = split_to_n_pieces_with_train_val(df_keeped,
                                              n = 1,
                                              train_ratio = args.train_ratio,
                                              seed = args.split_seed)
    train_df, val_df = pieces[0][0], pieces[0][1]
    print('====> with train ratio: ', args.train_ratio)
    print('====> train samples: ', len(train_df))
    print('====> val   samples: ', len(val_df))

    save_path = os.path.join(args.save_dir, 'ExpW_train.lst')
    train_df.to_csv(save_path, index=0, columns=['fname', 'id', 'y0', 'x0', 'x1', 'y1', 'conf', 'label'])
    save_path = os.path.join(args.save_dir, 'ExpW_val.lst')
    val_df.to_csv(save_path, index=0, columns=['fname', 'id', 'y0', 'x0', 'x1', 'y1', 'conf', 'label'])

    print('====> saved in {}'.format(args.save_dir))

if __name__ =='__main__':
    main()