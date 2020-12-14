import os
import  tqdm
import  torch
import  numpy as np
from  fer_pytorch.utils.logger import TxtLogger
from  fer_pytorch.config.default_cfg import  get_fer_cfg_defaults
from  fer_pytorch.datasets import  get_fer_test_dataloader
from  fer_pytorch.models.build_model import  build_model
from  fer_pytorch.utils.common import setup_seed,create_dir_maybe
from sklearn.metrics import classification_report,accuracy_score


import  argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=os.path.dirname(__file__)+'/../configs/mobilev2_head_stage2.yaml',
                        type=str,
                        help="")
    parser.add_argument("--cp_err_images_to_dir",
                        default='/home/lhw/m2_disk/data/DataForStage2_label_model_err/',
                        type=str,
                        help="")
    return parser.parse_args()

def val(model, val_dataloader):
    all_preds  = None
    all_labels = None
    all_paths  = []
    model.eval()
    for batch in tqdm.tqdm(val_dataloader):
        with torch.no_grad():
            imgs = batch[0].cuda()
            labels = batch[1].cuda()
            paths = batch[2]
            pred_loggits = model(imgs)
            labels = labels.squeeze()
        if all_preds is None:
            all_preds = pred_loggits.detach().cpu().numpy()
            all_labels = labels.detach().cpu().numpy()
            all_paths = list(paths)
        else:
            all_preds = np.append(all_preds, pred_loggits.detach().cpu().numpy(), axis=0)
            all_labels = np.append(all_labels, labels.detach().cpu().numpy(), axis=0)
            all_paths.extend(list(paths))

    all_preds = np.argmax(all_preds, axis=1)
    print("all preds shape: ", all_preds.shape)
    print("all labels shape: ", all_labels.shape)
    print("all paths len: ", len(all_paths))
    acc = accuracy_score(y_pred= all_preds, y_true= all_labels)
    print('acc: ', acc)
    print(classification_report(y_pred=all_preds, y_true=all_labels))
    err_indexes = np.where(all_preds != all_labels)[0]
    print('err num: ', len(err_indexes))

    save_dir = cfg.TEST.cp_err_images_to_dir
    for inx in err_indexes:
        src_path = all_paths[inx]
        sub_dir = 'pred_{}'.format(int(all_preds[inx]))
        create_dir_maybe(save_dir+'/'+sub_dir)
        dst_path = os.path.join(save_dir,sub_dir, os.path.basename(src_path))
        cmd = 'mv {} {}'.format(src_path, dst_path)
        print(cmd)
        os.system(cmd)

def infer(cfg):
    data_loader = get_fer_test_dataloader(cfg)
    model = build_model(cfg)
    model = model.cuda()
    model.load_state_dict(torch.load(cfg.TEST.model_load_path))
    val(model, data_loader)


if __name__ == '__main__':
    setup_seed(666)
    cfg = get_fer_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ',args.config.strip())
    cfg.merge_from_file(args.config)
    cfg.TEST.cp_err_images_to_dir = args.cp_err_images_to_dir
    print(cfg)
    infer(cfg)