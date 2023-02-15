import argparse
import nibabel
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
import torch
import numpy as np
from torch import nn
import time
import os
import moxing as mox
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--pretrain_url', type=str, default=None, help='Pretrain')
args_opt, unparsed  = parser.parse_known_args()
local_data_url = '/cache/data'
local_train_url = '/cache/ckpt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = args_opt.data_url
checkpoint_pretrain_resnet_10_23dataset = args_opt.pretrain_url
input_D = 128
input_H = 92
input_W = 128
num_seg_classes = 3
mox.file.copy_parallel(args_opt.data_url, local_data_url)

image = []
uurl = os.path.join(data + 'images')
img_lists = os.listdir(uurl)
label = os.path.join(data + 'labels/label.csv')
labell = pd.read_csv(label)
labels = []
for idx in range(0, len(img_lists)):
    im = nibabel.load(os.path.join(uurl + '/' + str(img_lists[idx])))
    im = np.array(im.get_fdata(), dtype=np.float32)
    image.append(im)
    s = img_lists[idx].split('.')[0]
    t = np.array(labell[labell['Subject ID'] == s]['Type'])
    labels.append(t[0])

outurl = local_train_url
criterion = nn.CrossEntropyLoss()
def save_checkpoint(epochs,optimizer,model,filepath):
    checkpoint = {'epochs':epochs,
                  'optimizer_state_dict':optimizer.state_dict(),
                  'model_state_dict':model.state_dict()}
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    res = os.path.join(filepath, 'checkpoint_model_epoch_{}.pth'.format(epochs))
    torch.save(checkpoint,res)

def train_data(model, train_dataloaders, valid_dataloaders, epochs, optimizer, scheduler, criterion, checkpoint_path,
               device='cpu'):
    start = time.time()
    model_indicators = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'train_f1_score', 'val_loss', 'val_acc', 'val_f1_score'])
    steps = 0
    n_epochs_stop = 10
    min_val_f1_score = 0
    epochs_no_improve = 0
    model.to(device)
    for e in range(epochs):
        model.train()
        train_loss = 0

        train_acc = 0

        train_correct_sum = 0
        train_simple_cnt = 0

        train_f1_score = 0

        y_train_true = []
        y_train_pred = []
        for i, (images, labels) in enumerate(train_dataloaders):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct_sum += (labels.data == train_predicted).sum().item()
            train_simple_cnt += labels.size(0)
            y_train_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())

        scheduler.step()

        val_acc = 0

        val_correct_sum = 0
        val_simple_cnt = 0
        val_loss = 0

        val_f1_score = 0

        y_val_true = []
        y_val_pred = []
        with torch.no_grad():
            model.eval()
            for i, (images, labels) in enumerate(valid_dataloaders):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, val_predicted = torch.max(outputs.data, 1)
                val_correct_sum += (labels.data == val_predicted).sum().item()
                val_simple_cnt += labels.size(0)
                y_val_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().detach().numpy())).tolist())

        train_loss = train_loss / len(train_dataloaders)
        val_loss = val_loss / len(valid_dataloaders)
        train_acc = train_correct_sum / train_simple_cnt
        val_acc = val_correct_sum / val_simple_cnt
        train_f1_score = f1_score(y_train_true, y_train_pred, average='macro')
        val_f1_score = f1_score(y_val_true, y_val_pred, average='macro')
        print('Epochs: {}/{}...'.format(e + 1, epochs),
              'Trian Loss:{:.3f}...'.format(train_loss),
              'Trian Accuracy:{:.3f}...'.format(train_acc),
              'Trian F1 Score:{:.3f}...'.format(train_f1_score),
              'Val Loss:{:.3f}...'.format(val_loss),
              'Val Accuracy:{:.3f}...'.format(val_acc),
              'Val F1 Score:{:.3f}'.format(val_f1_score))
        model_indicators.loc[model_indicators.shape[0]] = [e, train_loss, train_acc, train_f1_score, val_loss, val_acc,
                                                           val_f1_score]
        if val_f1_score >= min_val_f1_score:
            save_checkpoint(e + 1, optimizer, model, checkpoint_path)
            epochs_no_improve = 0
            min_val_f1_score = val_f1_score
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')

    plt_result(model_indicators)

    end = time.time()
    runing_time = end - start
    print('Training time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))
    mox.file.copy_parallel(local_train_url, args_opt.train_url)


def plt_result(dataframe):
    fig = plt.figure(figsize=(16, 5))

    fig.add_subplot(1, 3, 1)
    plt.plot(dataframe['epoch'], dataframe['train_loss'], 'b-', label='Train loss')
    plt.plot(dataframe['epoch'], dataframe['val_loss'], 'r-', label='Val loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    fig.add_subplot(1, 3, 2)
    plt.plot(dataframe['epoch'], dataframe['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(dataframe['epoch'], dataframe['val_acc'], 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    fig.add_subplot(1, 3, 3)
    plt.plot(dataframe['epoch'], dataframe['train_f1_score'], 'b-', label='Train F1 Score')
    plt.plot(dataframe['epoch'], dataframe['val_f1_score'], 'r-', label='Val F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()

    plt.show()
