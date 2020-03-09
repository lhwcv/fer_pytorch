from __future__ import print_function
import numpy as np
import h5py
from fer_pytorch.datasets.image_list_dataset import ImageList_Dataset


class FER2013_Dataset(ImageList_Dataset):
    """
    FER2013 Dataset.
    """
    def __init__(self, cfg, is_train=True):
        label_path = cfg.DATA.train_label_path if is_train else \
            cfg.DATA.val_label_path

        self.is_train = is_train
        self.data = h5py.File(label_path, 'r', driver='core')
        self.imgs = self.data['pixel']
        self.labels = self.data['label']
        self.imgs = np.asarray(self.imgs)
        self.imgs = self.imgs.reshape((self.imgs.size//(48*48), 48, 48))

        ## final choose by  cfg.DATA.wanted_catogories
        self.imgs3channels = []
        self.labels_needed = []

        print('===> orginal samples: ', len(self.imgs))

        ## re map label, and maybe drop some
        self.build_label_mapping_fn(cfg.DATA.wanted_catogories)
        print('===> wanted: ', cfg.DATA.wanted_catogories)
        for ks in cfg.DATA.wanted_catogories:
            for k in ks:
                v = self.label_mapping()[k]
                print('{}({})\t---- map to: {:>}'.format(k, v, self.map_fn(v)))

        for i, img in enumerate(self.imgs):
            label = self.map_fn(self.labels[i])
            if label==-1:
                continue
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            self.imgs3channels.append(img)
            self.labels_needed.append(label)
        print('===> filtered samples: ', len(self.labels_needed))
        del  self.labels
        del  self.imgs
        super(FER2013_Dataset, self).__init__(
            cfg,
            self.imgs3channels,
            self.labels_needed,
            is_train = is_train,
            read_img_from_file = False
        )

    def label_mapping(self):
        return {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "sad": 4,
            "surprised": 5,
            "neutral": 6
        }
