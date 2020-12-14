import  torch
import  pickle
import  cv2
import  random
import  numpy as np
import  math
from torch.utils.data import  DataLoader,Dataset
from albumentations import (
    Blur,GaussianBlur,MedianBlur,
    HueSaturationValue,
    RandomBrightnessContrast,
    Normalize,
    OneOf, Compose,
    NoOp,
)

MEAN_LANDMARK = np.array([
        0.2435, 0.3053, 0.3996, 0.3032, 0.5115, 0.3064, 0.6159, 0.2999, 0.7380,
        0.2975, 0.4218, 0.6247, 0.6028, 0.6202, 0.5181, 0.4753, 0.5136, 0.8485], np.float32)

class CZ_Head_Landmark(Dataset):
    def __init__(self, cfg, is_train):
        super(CZ_Head_Landmark, self).__init__()
        root_dir = cfg.DATA.label_dir
        self.infos = []
        if is_train:
            self.infos = pickle.load(open(root_dir+'train_infos.pkl', 'rb'))

        else:
            self.infos = pickle.load(open(root_dir + 'val_infos.pkl', 'rb'))

        self.aug_train = self._aug_train()
        self.aug_norm  = self._aug_test()
        self.cfg = cfg
        self.is_train = is_train
        print('samples: ', len(self.infos))

    def _crop_by_bbox_aug(self, img, landmarks, bbox):
        x, y, w, h = bbox
        new_w = w +  w * random.randint(4, 10) / 10   ## 1.4 -- 2.0
        new_h = h +  h * random.randint(4, 10) / 10  ## 1.4 -- 2.0

        if not self.is_train:
            new_w = w + w * 1.4
            new_h = h + h * 1.4
        x0 = x + w/2 - new_w/2
        y0 = y + h/2 - new_h/2
        x1 = x + w / 2   + new_w / 2
        y1 = y + h / 2   + new_h / 2

        #y1 biaz
        y1 = y1 - 0.2*h
        x0 = 0 if x0 < 0 else x0
        y0 = 0 if y0 < 0 else y0
        x1 = img.shape[1] - 1 if x1 > img.shape[1] - 1 else x1
        y1 = img.shape[0] - 1 if y1 > img.shape[0] - 1 else y1
        landmarks_crop = []
        img_crop = img[int(y0):int(y1), int(x0):int(x1)].copy()
        for p in landmarks:
            landmarks_crop.append([p[0]-x0, p[1]-y0 ])
        return  img_crop, landmarks_crop

    def _resize_and_norm_landmarks(self, img, landmarks):
        size = self.cfg.DATA.input_size
        landmarks_resized = []
        rw = size / img.shape[1]
        rh = size / img.shape[0]
        img_resized = cv2.resize(img, (size, size), cv2.INTER_CUBIC)
        for p in landmarks:
            landmarks_resized.append([p[0]*rw / size,  p[1]*rh / size])
        return img_resized, landmarks_resized

    def _aug_train(self):
        aug = Compose(
            [
                OneOf(
                    [
                        HueSaturationValue(hue_shift_limit=20,
                                           sat_shift_limit=20,
                                           val_shift_limit=20,
                                           p=0.5),
                        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)
                    ]
                )
            ],
            p=1.0)
        return aug

    def _aug_test(self):
        aug = Compose(
            [
                # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                Normalize(mean=(0.485, 0.456, 0.406), std=(1.0 / 255, 1.0 / 255, 1.0 / 255))
            ],
            p=1.0)
        return aug

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, inx):
        info = self.infos[inx]
        img_fn = info['img_full_path']
        bbox   = info['bounding_bbox']
        landmarks = info['landmarks']
        #norm_base_dis = info['norm_base_dis']

        img = cv2.imread(img_fn)
        assert img is not None, img_fn
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train:
            img = self.aug_train(image=img)['image']
            if random.random() > 0.2:
                img, landmarks = self._crop_by_bbox_aug(img, landmarks, bbox)
        else:
            img, landmarks = self._crop_by_bbox_aug(img, landmarks, bbox)

        img, landmarks = self._resize_and_norm_landmarks(img, landmarks)

        ##debug
        # gui = img.copy()
        # for p in landmarks:
        #     p[0] = p[0]* img.shape[1]
        #     p[1] = p[1] * img.shape[0]
        #     cv2.circle(gui, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        # for i in range(9):
        #     p = [0,1]
        #     p[0] = MEAN_LANDMARK[2*i] * img.shape[1]
        #     p[1] = MEAN_LANDMARK[2*i+1] * img.shape[0]
        #     cv2.circle(gui, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        # cv2.imwrite('/home/lhw/gui.jpg', gui)

        img = self.aug_norm(image=img)['image']
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)

        norm_base_dis = (landmarks[1][0] - landmarks[3][0]) * (landmarks[1][0] - landmarks[3][0]) + \
                        (landmarks[1][1] - landmarks[3][1]) * (landmarks[1][1] - landmarks[3][1])
        norm_base_dis = math.sqrt(norm_base_dis)

        landmarks = np.array(landmarks, np.float32).reshape(-1)
        landmarks = landmarks- MEAN_LANDMARK
        label = torch.from_numpy(landmarks)
        norm_base_dis = torch.FloatTensor([norm_base_dis])
        return img, label,norm_base_dis, img_fn


if __name__ == '__main__':
    import  tqdm
    from fer_pytorch.config.default_cfg import get_fer_cfg_defaults
    cfg = get_fer_cfg_defaults()
    cfg.DATA.label_dir =  '/home/lhw/data/FaceDataset/LS3D_W_CZUR_9_landmark/'
    cfg.DATA.input_size = 128
    dataset = CZ_Head_Landmark(cfg, is_train=True)
    all_label = torch.zeros((18), dtype=torch.float32)
    for img, label,dis, fn in tqdm.tqdm(dataset):
        # print(img.shape)
        # print(label.shape)
        print(dis)
        all_label += label
        exit()

    print('mean shape: ', all_label / len(dataset))
