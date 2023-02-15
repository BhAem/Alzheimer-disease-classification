import argparse
import os
import nibabel
import numpy as np
import pandas as pd
import torch
from MyDataSet import MyDataset
from torch.utils.data import DataLoader
import moxing as mox
from model import generate_model

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--pretrain_url', type=str, default=None, help='Pretrain')
args_opt, unparsed  = parser.parse_known_args()

local_data_url = '/cache/data'
local_ckpt_url = '/cache/ckpt'
test_dir = args_opt.data_url

cvs_url = 'predict_result.csv'
checkpoint_pretrain_resnet_10_23dataset = args_opt.pretrain_url

input_D = 128
input_H = 92
input_W = 128
num_seg_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(filepath, model_name, phase='train', device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    if model_name == 'medicanet_resnet3d_10':
        model, _ = generate_model(sample_input_W=input_W,
                                  sample_input_H=input_H,
                                  sample_input_D=input_D,
                                  num_seg_classes=num_seg_classes,
                                  phase=phase,
                                  pretrain_path=checkpoint_pretrain_resnet_10_23dataset)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(test_dataloader, loadmodel, device, result_path):
    result_df = pd.DataFrame(columns=['testa_id', 'AD','MCI','CN'])
    with torch.no_grad():
        loadmodel.to(device)
        loadmodel.eval()
        for ii, image in enumerate(test_dataloader):
            image = image.to(device)
            output = loadmodel(image)
            _, indexs = torch.max(output.data, 1)
            arr = np.array(torch.Tensor.cpu(output))
            sum = np.exp(arr[0][0]) + np.exp(arr[0][1]) + np.exp(arr[0][2])
            temp0 = format(np.exp(arr[0][0])/sum, '.4%')
            temp1 = format(np.exp(arr[0][1])/sum, '.4%')
            temp2 = format(np.exp(arr[0][2])/sum, '.4%')
            res = [temp0, temp1, temp2]
            res = np.array(res)
            print('test data {} result:'.format(ii+1))
            print('AD:'+ res[0], 'MCI:' + res[1], 'CN:' + res[2])
            result_df.loc[result_df.shape[0]] = [('testa_{}'.format(ii + 1)), res[0], res[1], res[2]]
            print('--' * 20)
    result_df.to_csv(result_path, index=False)

if __name__ == '__main__':
    if not os.path.exists(local_ckpt_url ):
        os.makedirs(local_ckpt_url )
    if args_opt.checkpoint_path:
        checkpoint_file = os.path.join(local_ckpt_url, os.path.split(args_opt.checkpoint_path)[1])
    mox.file.copy_parallel(args_opt.data_url, local_data_url)
    mox.file.copy_parallel(args_opt.checkpoint_path, checkpoint_file)
    test_image = []
    testdir = test_dir
    test_dir = os.listdir(test_dir)
    for idx in range(0, len(test_dir)):
        im = nibabel.load(testdir + '/' + test_dir[idx])
        im = np.array(im.get_fdata(), dtype=np.float32)
        test_image.append(im)

    test_datasets = MyDataset(datas=test_image, shape=3, input_D=input_D, input_H=input_H,
                              input_W=input_W, phase='test')
    test_loader = DataLoader(dataset=test_datasets)
    loadmodel = load_checkpoint(checkpoint_file, 'medicanet_resnet3d_10', 'test',
                                     device)
    path = os.path.join(args_opt.train_url, cvs_url)
    predict(test_loader, loadmodel, device, path)
