import  os
import cv2
import  torch
from torch.utils.data import  DataLoader,Dataset
from fer_pytorch.datasets.aug import fer_test_aug,fer_train_aug
from fer_pytorch.datasets.base import FER_Dataset

class ImageList_Dataset(FER_Dataset):
    def __init__(self,
                 cfg,
                 img_path_list,
                 labels_list = None,
                 is_train = False,
                 xyxy = None):
        super(ImageList_Dataset, self).__init__()
        self.cfg = cfg
        self.img_path_list = img_path_list
        self.labels_list = labels_list
        self.xyxy = xyxy
        if is_train:
            self.aug = fer_train_aug(cfg.DATA.input_size,
                                     cfg.DATA.crop_residual_pix)
        else:
            self.aug = fer_test_aug(cfg.DATA.input_size)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        path = self.img_path_list[idx]
        path = os.path.join(self.cfg.DATA.img_dir, path)
        assert os.path.exists(path), path
        img = cv2.imread(path)
        if self.xyxy is not  None:
            x0,y0,x1,y1 = self.xyxy[0][idx], self.xyxy[1][idx],\
                          self.xyxy[2][idx], self.xyxy[3][idx]
            img = img[y0 : y1, x0 : x1, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.aug(image = img)['image']
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)
        if  self.labels_list is not None:
            label = torch.LongTensor([self.labels_list[idx]])
        else:
            label = torch.LongTensor([0])
        return  img, label