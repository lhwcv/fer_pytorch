from  fer_pytorch.datasets.image_list_dataset import  ImageList_Dataset
import pandas as pd

class ExpW_Dataset(ImageList_Dataset):
    def __init__(self,cfg, is_train = True):
        label_path = cfg.DATA.train_label_path if is_train else \
            cfg.DATA.val_label_path
        df = pd.read_csv(label_path)
        print('===> orginal samples: ', len(df))
        self.build_label_mapping_fn(cfg.DATA.wanted_catogories)
        df['label'] = df['label'].map(self.map_fn)
        df = df[df['label'] != -1]
        print('===> wanted: ', cfg.DATA.wanted_catogories)

        for ks in cfg.DATA.wanted_catogories:
            for k in ks:
                v = self.label_mapping()[k]
                print('{}({})\t---- map to: {:>}'.format(k, v, self.map_fn(v)))
        print('===> filtered samples: ', len(df))

        img_path_list = df['fname'].to_list()
        labels_list = df['label'].to_list()
        x0 = df['x0'].to_list()
        y0 = df['y0'].to_list()
        x1 = df['x1'].to_list()
        y1 = df['y1'].to_list()
        super(ExpW_Dataset, self).__init__(
            cfg,
            img_path_list,
            labels_list,
            is_train = is_train,
            xyxy = [x0, y0, x1, y1]
        )

    def label_mapping(self):
        return {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "sad": 4,
            "surprise": 5,
            "neutral": 6
        }

if __name__ == '__main__':
    label_path = './cache/ExpW_train.lst'
    from fer_pytorch.config.default_cfg import get_fer_cfg_defaults
    cfg = get_fer_cfg_defaults()
    dataset  = ExpW_Dataset(cfg, label_path)
    for img, label in dataset:
        print(img.shape)
        exit()



