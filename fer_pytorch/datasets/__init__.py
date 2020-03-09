from  torch.utils.data import  Dataset,DataLoader
from fer_pytorch.datasets.ExpW import  ExpW_Dataset
from fer_pytorch.datasets.FER2013 import  FER2013_Dataset
#from  fer_pytorch.datasets.AffectNet import  AffectNet_Dataset

__mapping_dataset = {
    'ExpW': ExpW_Dataset,
    'FER2013':FER2013_Dataset
}

def get_dataset(cfg, is_train = True):
    if cfg.DATA.dataset_type not in __mapping_dataset.keys():
        raise  NotImplementedError('Dataset Type not supported!')
    return  __mapping_dataset[cfg.DATA.dataset_type](
        cfg,
        is_train = is_train
    )

def get_fer_train_dataloader(cfg):
    ds = get_dataset(cfg, True)
    dloader = DataLoader(
        ds,
        batch_size = cfg.TRAIN.batch_size,
        shuffle = True,
        num_workers = cfg.SYSTEM.NUM_WORKERS,
        drop_last=True
    )
    return  dloader

def get_fer_val_dataloader(cfg):
    ds = get_dataset(cfg, False)
    dloader = DataLoader(
        ds,
        batch_size = cfg.TRAIN.batch_size,
        shuffle = False,
        num_workers = cfg.SYSTEM.NUM_WORKERS,
        drop_last=True
    )
    return  dloader

def get_fer_test_dataloader(cfg):
    ds = get_dataset(cfg, False)
    dloader = DataLoader(
        ds,
        batch_size = cfg.TEST.batch_size,
        shuffle = False,
        num_workers = cfg.SYSTEM.NUM_WORKERS
    )
    return dloader


