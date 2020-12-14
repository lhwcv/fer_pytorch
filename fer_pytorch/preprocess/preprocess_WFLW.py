import  cv2
import  os
import  tqdm
import  numpy as np
from fer_pytorch.utils.common import  create_dir_maybe
import  json


def get_9pts(landmark106):
    d = []
    d.append(landmark106[0])
    d.append(landmark106[96])
    d.append(landmark106[51])
    d.append(landmark106[97])
    d.append(landmark106[32])

    d.append(landmark106[54])
    d.append(landmark106[16])
    d.append(landmark106[76])
    d.append(landmark106[82])
    return d

def save_to_labelme_format(img, img_fn, bboxes, landmarks, with_gui=False):
    # ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
    json_content = {
        'version': '4.5.6',
        'flags': {},
        'shapes': [],
        'imagePath': os.path.basename(img_fn),
        'imageData': None,
        'imageHeight': img.shape[0],
        'imageWidth': img.shape[1],
    }
    for r, pts in zip(bboxes, landmarks):
        # for r in new_bboxs:
        item = {
            "label": "head",
            "points": [
                [
                    r[0],
                    r[1],
                ],
                [
                    r[2],
                    r[3],
                ]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        json_content['shapes'].append(item)
        if with_gui:
            cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0,0,255),3,16)
            for x, y in pts:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        for p_idx, p in enumerate(pts):
            item = {
                "label": "landmark_{:02d}".format(p_idx),
                "points": [
                    [
                        np.float(p[0]),
                        np.float(p[1]),
                    ]
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": {}
            }
            json_content['shapes'].append(item)
    if with_gui:
        save_fn = img_fn.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        save_fn = save_fn + '_gui.jpg'
        cv2.imwrite(save_fn,img)

    if len(json_content['shapes']) > 0:
        save_fn = img_fn.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        save_fn = save_fn + '.json'
        json.dump(json_content, open(save_fn, 'w'), indent=4)

def convert_to_9pts_label_me(img_dir, label_fn, img_label_save_dir, phase='train'):
    contents = []
    with open(label_fn, 'r') as f:
        cc = f.readlines()
        for c in cc:
            c = c.strip()
            c = c.split(' ')
            assert  len(c) == 207
            item = {
                'landmark': np.array(list(map(float, c[:196]))).reshape(-1,2),
                'bbox': list(map(float, c[196:200]) ),
                'fname': c[-1]
            }
            contents.append(item)
    counter = 0
    for c in tqdm.tqdm(contents):
        landmark = get_9pts(c['landmark'])
        b = c['bbox']
        img_fn = img_dir + '/' + c['fname']
        img = cv2.imread(img_fn)
        h,w,_ = img.shape
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        bw = bw * 1.6
        bh = bh * 1.6
        xmin = (b[0] + b[2])/2 - 0.5*bw
        xmax = (b[0] + b[2])/2 + 0.5*bw
        ymin = (b[1] + b[3]) / 2 - 0.5 * bh
        ymax = (b[1] + b[3]) / 2 + 0.5 * bh
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] - 1 if xmax > img.shape[1] else xmax
        ymax = img.shape[0] - 1 if ymax > img.shape[0] else ymax

        landmark_adjust = []
        for p in landmark:
            landmark_adjust.append([p[0]-xmin, p[1] - ymin])
        sub_im = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()

        b[0] = b[0] - xmin
        b[2] = b[2] - xmin
        b[1] = b[1] - ymin
        b[3] = b[3] - ymin
        basename = os.path.basename(img_fn).replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        save_fn = os.path.join(img_label_save_dir, "{}_{}_id_{}.jpg".format(phase,basename, counter))
        cv2.imwrite(save_fn, sub_im)
        save_to_labelme_format(sub_im, save_fn,[b],[landmark_adjust],with_gui=False )
        counter+=1

def main():
    anno_dir = '/home/lhw/data/FaceDataset/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/'
    img_dir  = '/home/lhw/data/FaceDataset/WFLW/WFLW_images/'
    save_dir = '/home/lhw/data/FaceDataset/WFLW/WFLW_labelme/'
    create_dir_maybe(save_dir)
    train_fn = anno_dir+'/list_98pt_rect_attr_train.txt'
    test_fn  = anno_dir+'/list_98pt_rect_attr_test.txt'

    convert_to_9pts_label_me(img_dir, train_fn, save_dir,phase='train')
    convert_to_9pts_label_me(img_dir, test_fn, save_dir,phase='test')

if __name__ == '__main__':
    main()