import cv2
import onnx
import torch
import argparse
import os
import  numpy  as np
from  fer_pytorch.models.build_model import  build_model
from  fer_pytorch.config import  get_fer_cfg_defaults

def get_args():
    import  os
    dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=dir + '/../configs/mobilev2_head_stage2.yaml',
                        type=str,
                        help="")
    return parser.parse_args()

def main(cfg):
    dummy_input = torch.randn(1, 3, cfg.DATA.input_size,  cfg.DATA.input_size, device='cuda')
    model = build_model(cfg).cuda()

    input_names = [ "input" ]
    output_names = [ "output" ]
    print('model: ',cfg.TEST.model_load_path )
    #model.load_state_dict(torch.load(cfg.TEST.model_load_path),strict=True)
    output_path = cfg.TEST.model_load_path.replace('pth', 'onnx')
    torch.onnx.export(model, dummy_input,output_path,
                      export_params=True,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      )#keep_initializers_as_inputs=True)
    model = onnx.load(output_path)
    onnx.checker.check_model(model)



if __name__ == '__main__':
    cfg = get_fer_cfg_defaults()
    args = get_args()
    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ',args.config.strip())
    cfg.merge_from_file(args.config)
    main(cfg)
