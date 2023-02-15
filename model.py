'''
@article{chen2019med3d,
 title = {Med3D: Transfer Learning for 3D Medical Image Analysis},
 author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
 journal={arXiv preprint arXiv:1904.00625},
 year = {2019}
}
'''
import torch
from torch import nn
from models import resnet


def generate_model(sample_input_W, sample_input_H, sample_input_D, num_seg_classes=3, no_cuda=False, phase='train', pretrain_path=None, new_layer_names=['avgpool','fc']):
    model = resnet.resnet10(
        sample_input_W=sample_input_W,
        sample_input_H=sample_input_H,
        sample_input_D=sample_input_D,
        num_seg_classes=num_seg_classes)

    if not no_cuda:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    if phase != 'test' and pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
