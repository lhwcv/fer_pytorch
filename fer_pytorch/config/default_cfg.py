from yacs.config import CfgNode as CN
__all_ = ['get_fer_cfg_defaults']

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 2

_C.DATA = CN()
_C.DATA.input_size = 256
_C.DATA.crop_residual_pix = 16
_C.DATA.dataset_type = 'ExpW'
_C.DATA.img_dir = '/home/lhw/yangzhi/FaceExpRecog/2.data/ExpW/image/origin/'
_C.DATA.label_dir = '/home/lhw/yangzhi/FaceExpRecog/2.data/ExpW/label/'
_C.DATA.train_label_path = _C.DATA.label_dir+'/label_filtered_conf_thresh_0_no_0_2_part_0_train.lst'
_C.DATA.val_label_path =  _C.DATA.label_dir+'/label_filtered_conf_thresh_0_no_0_2_part_0_val.lst'
_C.DATA.need_crop_xyxy = False
_C.DATA.wanted_catogories = [ ['neutral'],['happy'], ['sad'],['angry']]

_C.TRAIN = CN()
_C.TRAIN.do_train = True
_C.TRAIN.batch_size = 64
_C.TRAIN.save_dir = '/home/lhw/data_disk_fast/comp_workspace/saved_model/fer_seres50/'
_C.TRAIN.gradient_accumulation_steps = 1
_C.TRAIN.num_train_epochs = 30
_C.TRAIN.warmup_steps = 500
_C.TRAIN.learning_rate = 8*1e-5
_C.TRAIN.milestones = [2000, 4000]
_C.TRAIN.lr_decay_gamma = 0.2
_C.TRAIN.weight_decay = 0.0
_C.TRAIN.device_ids_str = "0"
_C.TRAIN.device_ids = [0]
_C.TRAIN.log_steps = 200
_C.TRAIN.adam_epsilon = 1e-6
_C.TRAIN.early_stop_n = 4
_C.TRAIN.device_ids_str = "0"
_C.TRAIN.device_ids = [0]

_C.TEST = CN()
_C.TEST.do_train = False
_C.TEST.batch_size = 256
_C.TEST.model_load_path = '/home/lhw/data_disk_fast/comp_workspace/saved_model/fer_mobile_v2_data_part0/'
_C.TEST.save_dir = '/home/lhw/data_disk_fast/comp_workspace/saved_model/test_log/'
_C.TEST.device_ids_str = "0"
_C.TEST.device_ids = [0]
_C.TEST.cp_err_images_to_dir =''

_C.MODEL = CN()
_C.MODEL.model_name = 'se_resnext50_32x4d'
_C.MODEL.num_classes = 4

def get_fer_cfg_defaults(merge_from = None):
  cfg =  _C.clone()
  if merge_from is not None:
      cfg.merge_from_other_cfg(merge_from)
  return cfg
