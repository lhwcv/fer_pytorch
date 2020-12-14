import cv2
import  glob
import  tqdm
import  os
import  random
import  json
from fer_pytorch.utils.common import create_dir_maybe
import  pickle
import  numpy as np
import  math

def get_img_fn_by_ann_fn(ann_fn):
    #print(ann_fn)
    img_fn = ann_fn.replace('.json','.jpg')
    if not os.path.exists(img_fn):
        img_fn = ann_fn.replace('.json', '.jpeg')
    if not os.path.exists(img_fn):
        img_fn = ann_fn.replace('.json', '.png')

    if not os.path.exists(img_fn):
        img_fn = ann_fn.replace('.xml', '.jpeg')
    if not os.path.exists(img_fn):
        img_fn = ann_fn.replace('.xml', '.png')
    if not os.path.exists(img_fn):
            img_fn = ann_fn.replace('.xml', '.jpg')
    assert  os.path.exists(img_fn), img_fn
    return img_fn

def save_to_pickle(json_files, save_fn):
    infos = []
    for fn in tqdm.tqdm(json_files):
        c = json.load(open(fn,'r'))
        full_img_path = os.path.dirname(fn)+'/'+c['imagePath']
        landmarks = []
        landmark_map = {}
        for s in c['shapes']:
            landmark_map[s['label'] ] = s['points'][0]
        for i in range(9):
            key = "landmark_{:02d}".format(i)
            landmarks.append(landmark_map[key])

        norm_base_dis = (landmarks[1][0] - landmarks[3][0])*(landmarks[1][0] - landmarks[3][0])+\
                        (landmarks[1][1] - landmarks[3][1])*(landmarks[1][1] - landmarks[3][1])
        norm_base_dis = math.sqrt(norm_base_dis)
        item={
            'img_full_path': full_img_path,
            'landmarks':landmarks,
            'bounding_bbox': cv2.boundingRect(np.array(landmarks,np.float32)),
            'norm_base_dis': norm_base_dis
        }
        infos.append(item)
    print('infos len: ',len(infos))
    print('save to: ', save_fn)
    pickle.dump(infos, open(save_fn,'wb'))

def main():
    data_dir = '/home/lhw/data/FaceDataset/LS3D_W_CZUR_9_landmark/'
    all_files = glob.glob(data_dir + '/**//*.json')
    print('all samples: ',len(all_files))
    random.seed(666)
    N = int(len(all_files) * 0.8)
    train_files = all_files[:N]
    val_files  = all_files[N:]
    save_to_pickle(train_files, data_dir+'train_infos.pkl')
    save_to_pickle(val_files, data_dir + 'val_infos.pkl')

if __name__ == '__main__':
    main()