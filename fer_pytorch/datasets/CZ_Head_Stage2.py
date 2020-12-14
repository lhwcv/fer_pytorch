import  torch
from fer_pytorch.datasets.image_list_dataset import ImageList_Dataset

class CZ_Head_Stage2(ImageList_Dataset):
    def __init__(self, cfg, is_train):
        root_dir = cfg.DATA.label_dir
        self.imgs = []
        self.labels = []
        if is_train:
            with open(root_dir+'/train_pos.txt','r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    self.imgs.append(line)
                    self.labels.append(1)
            with open(root_dir+'/train_neg.txt','r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    self.imgs.append(line)
                    self.labels.append(0)

        else:
            with open(root_dir + '/val_pos.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    self.imgs.append(line)
                    self.labels.append(1)
            with open(root_dir + '/val_neg.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    self.imgs.append(line)
                    self.labels.append(0)
        print('images: ', len(self.imgs))

        super(CZ_Head_Stage2, self).__init__(
            cfg,
            self.imgs,
            self.labels,
            is_train=is_train,
            read_img_from_file=True
        )

    def label_mapping(self):
        return  None

