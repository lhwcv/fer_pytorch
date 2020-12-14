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
import  cv2
from  albumentations import  Normalize,Compose
from  fer_pytorch.datasets.CZ_Head_landmark import MEAN_LANDMARK
from mmdet.apis import inference_detector, init_detector
import  argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=os.path.dirname(__file__)+'/../configs/mobilev2_head_landmark9.yaml',
                        type=str,
                        help="")
    return parser.parse_args()



def load_detect_model():
    device = torch.device('cuda:0')
    config = '/home/lhw/data_disk_fast/czcv.haowei/ref/mmdetection/configs/cz_head/mobilenet_v3_modify_yolo_like.py'
    checkpoint = '/home/lhw/m2_disk/work_dir/mobilev3_modify/latest.pth'
    model = init_detector(config, checkpoint, device=device)
    return  model

def detect_max_head(model, img_fn, score_thres=0.7):
    head = []
    result = inference_detector(model, img_fn)
    all_dets = []
    for r in result:
        for d in r:
            if d[-1] > score_thres:
                all_dets.append([np.float(d[0]),np.float(d[1]),np.float(d[2]),np.float(d[3])])
    if len(all_dets)>0:
        max_head = [0,0,0,0]
        for d in all_dets:
            if abs(d[2] -d[0])*abs(d[3]-d[1]) > \
                abs(max_head[2] -max_head[0])*abs(max_head[3]-max_head[1]):
                max_head = d
        head.append(max_head)
        return  head
    else:
        return  head

def infer_one_and_draw(model, detect_model, frame):
    aug = Compose(
        [
            Normalize(mean=(0.485, 0.456, 0.406), std=(1.0 / 255, 1.0 / 255, 1.0 / 255))
        ],
        p=1.0)

    #frame = cv2.imread('/home/lhw/test.jpg')
    heads = detect_max_head(detect_model, frame)
    if len(heads) <1:
        return frame
    d = heads[0]
    xmin = d[0]
    ymin = d[1]
    xmax = d[2]
    ymax = d[3]
    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    scale = 0.2
    xmin = xmin - w * scale
    xmax = xmax + w * scale
    ymin = ymin - h * scale
    ymax = ymax + h * scale

    xmin = 0 if xmin < 0 else xmin
    ymin = 0 if ymin < 0 else ymin
    xmax = frame.shape[1] - 1 if xmax > frame.shape[1] else xmax
    ymax = frame.shape[0] - 1 if ymax > frame.shape[0] else ymax
    sub_im = frame[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
    img = cv2.cvtColor(sub_im, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (72, 72), cv2.INTER_CUBIC)
    img = aug(image=img)['image']
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).cuda().unsqueeze(0)
    pred = model(img).detach().cpu().numpy()[0]
    pred += MEAN_LANDMARK
    img = sub_im
    for i in [1,3,7,8]:
        p = [0, 1]
        p[0] = pred[2 * i] * img.shape[1] + xmin
        p[1] = pred[2 * i + 1] * img.shape[0] + ymin
        cv2.circle(frame, (int(p[0]), int(p[1])), 7, (0, 255, 0), -1)
    xmin = d[0]
    ymin = d[1]
    xmax = d[2]
    ymax = d[3]
    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255),3,16)
    return  frame

def infer(cfg):
    model = build_model(cfg)
    model = model.cuda()
    model.load_state_dict(torch.load(cfg.TEST.model_load_path))
    model.eval()
    detect_model = load_detect_model()
    camera = cv2.VideoCapture(0)
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, frame = camera.read()
        frame = infer_one_and_draw(model, detect_model, frame)
        cv2.namedWindow('det', 0)
        cv2.imshow('det', frame)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break


if __name__ == '__main__':
    setup_seed(666)
    cfg = get_fer_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ',args.config.strip())
    cfg.merge_from_file(args.config)
    print(cfg)
    infer(cfg)