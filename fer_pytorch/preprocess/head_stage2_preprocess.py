import cv2
import  glob
import  tqdm
import  os
import  random
import xml.etree.ElementTree as ET
from fer_pytorch.utils.common import create_dir_maybe

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


def gen_some_positive():
    data_root = '/home/lhw/m2_disk/data/CZUR_HEAD_GEN/'
    anno_dir = '/home/lhw/m2_disk/data/CZUR_HEAD_GEN/Annotations/'
    image_dir = '/home/lhw/m2_disk/data/CZUR_HEAD_GEN/JPEGImages/'
    save_directory = '/home/lhw/m2_disk/data/DataForStage2_label/gen/images/pos/'
    create_dir_maybe(save_directory)
    fns = os.listdir(anno_dir)
    scale = 0.05
    min_size = 40* 40
    for fn in tqdm.tqdm(fns):
        img_name = get_img_fn_by_ann_fn(image_dir + '/' + fn)
        img = cv2.imread(img_name)
        tree = ET.parse(anno_dir + '/' + fn)
        root = tree.getroot()
        img_name = os.path.basename(img_name).split('.')[0]
        cc = 0
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)
            w = abs(xmax - xmin)
            h = abs(ymax - ymin)
            if w * h > min_size:
                xmin = xmin - w * scale
                xmax = xmax + w * scale
                ymin = ymin - h * scale
                ymax = ymax + h * scale
                if random.random()<0.2 and w*h > 80*80:
                    ymax = ymax - h * random.randint(0,5)/10.0

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = img.shape[1] - 1 if xmax > img.shape[1] else xmax
                ymax = img.shape[0] - 1 if ymax > img.shape[0] else ymax
                sub_im = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
                save_fn = save_directory + '/{}_{}.jpg'.format(img_name, cc)
                cc+=1
                cv2.imwrite(save_fn, sub_im)

def main():
    data_dir = '/home/lhw/m2_disk/data/DataForStage2_label/'
    all_neg_files = glob.glob(data_dir + '/**/images/neg/*.jpg')
    all_pos_files = glob.glob(data_dir + '/**/images/pos/*.jpg')

    print('all negtive samples: ',len(all_neg_files))
    print('all positive samples: ', len(all_pos_files))
    random.seed(666)
    random.shuffle(all_neg_files)
    random.shuffle(all_pos_files)

    N = int(len(all_neg_files) * 0.8)
    train_neg_files = all_neg_files[:N]
    val_neg_files   = all_neg_files[N: ]

    N = int(len(all_pos_files) * 0.8)
    train_pos_files = all_pos_files[:N]
    val_pos_files = all_pos_files[N:]
    with open(data_dir+'/train_neg.txt','w') as f:
        for fn in train_neg_files:
            if '副本' not  in fn:
                f.write(fn+'\n')
    with open(data_dir+'/val_neg.txt','w') as f:
        for fn in val_neg_files:
            if '副本' not  in fn:
                f.write(fn+'\n')
    with open(data_dir+'/train_pos.txt','w') as f:
        for fn in train_pos_files:
            if '副本' not  in fn:
                f.write(fn+'\n')
    with open(data_dir+'/val_pos.txt','w') as f:
        for fn in val_pos_files:
            if '副本' not  in fn:
                f.write(fn+'\n')


if __name__ == '__main__':
    #gen_some_positive()
    main()