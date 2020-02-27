from  fer_pytorch.utils.logger import TxtLogger
from  fer_pytorch.config.default_cfg import  get_fer_cfg_defaults

from  fer_pytorch.datasets import  get_fer_train_dataloader,get_fer_val_dataloader

from  fer_pytorch.models.build_model import  build_model

from  fer_pytorch.loss.softmaxloss import  CrossEntropyLabelSmooth
from  fer_pytorch.optim.optimizer.nadam import Nadam
from  fer_pytorch.optim.lr_scheduler import WarmupMultiStepLR

from tools.simple_learner import  SimpleLearner

from  fer_pytorch.utils.common import setup_seed

import  argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default='./configs/expw_res50.yml',
                        type=str,
                        required=True,
                        help="")
    return parser.parse_args()

def train(cfg):
    train_loader = get_fer_train_dataloader(cfg)
    val_loader   = get_fer_val_dataloader(cfg)
    model = build_model(cfg)
    model = model.cuda()

    loss_fn = CrossEntropyLabelSmooth(num_classes = cfg.MODEL.num_classes)
    optimizer = Nadam(params=model.parameters(),lr=cfg.TRAIN.learning_rate)
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones= cfg.TRAIN.milestones,
        gamma = cfg.TRAIN.lr_decay_gamma
    )
    logger = TxtLogger(cfg.TRAIN.save_dir + "/logger.txt")
    learner =  SimpleLearner(
        model = model,
        loss_fn  = loss_fn,
        optimizer = optimizer,
        scheduler = lr_scheduler,
        logger = logger,
        save_dir = cfg.TRAIN.save_dir,
        log_steps = cfg.TRAIN.log_steps,
        device_ids = cfg.TRAIN.device_ids,
        gradient_accum_steps = 1,
        max_grad_norm = 1.0,
        batch_to_model_inputs_fn = None,
        early_stop_n= cfg.TRAIN.early_stop_n)

    learner.train(train_loader, val_loader)


if __name__ == '__main__':
    setup_seed(666)
    cfg = get_fer_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ',args.config.strip())
    cfg.merge_from_file(args.config)
    print(cfg)
    train(cfg)