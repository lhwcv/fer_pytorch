import  torch
from  fer_pytorch.models.backbone.torchvision_models import  (
     resnet18, resnet34, resnet50, resnet101, resnet152
)
from  fer_pytorch.models.backbone.senet import (
    se_resnext50_32x4d
)

from  fer_pytorch.models.backbone.mobilenet_v2 import  mobilenet_v2


def modify_resnet_fc(model, n_class):
    in_features = model.last_linear.in_features
    model.last_linear = torch.nn.Linear(in_features, n_class)
    return  model

def build_model(cfg):
    model_name = cfg.MODEL.model_name
    num_classes = cfg.MODEL.num_classes
    if model_name == 'res18':
        m = resnet18()
        m = modify_resnet_fc(m, num_classes)
        return m
    if model_name == 'res34':
        m = resnet34()
        m = modify_resnet_fc(m, num_classes)
        return m
    if model_name == 'res50':
        m = resnet50()
        m = modify_resnet_fc(m, num_classes)
        return m


    if model_name == 'se_resnext50_32x4d':
        m =  se_resnext50_32x4d()
        m.last_linear = torch.nn.Linear(m.last_linear.in_features, num_classes)
        return  m

    if model_name == 'mobilenet_v2':
        m =  mobilenet_v2(num_classes = num_classes, pretrained=False)
        m.classifier = torch.nn.Linear(m.classifier.in_features, num_classes)
        return m

    raise  NotImplementedError('{} no such models'.format(model_name))


if __name__ == '__main__':
    from  fer_pytorch.config.default_cfg import  get_fer_cfg_defaults
    cfg  = get_fer_cfg_defaults()
    cfg.MODEL.model_name = 'se_resnext50_32x4d'
    model = build_model(cfg)

