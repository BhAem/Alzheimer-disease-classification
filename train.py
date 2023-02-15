import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import tools
from MyDataSet import MyDataset
from model import generate_model

train_data, test_data, train_labels, test_labels = train_test_split(tools.image, tools.labels, test_size=0.2,
                                            random_state=42, stratify=tools.labels)
train_datasets = MyDataset(datas=train_data, labels=train_labels, shape=3,input_D=tools.input_D,input_H=tools.input_H,input_W=tools.input_W,phase='train')
val_datasets = MyDataset(datas=test_data, labels=test_labels, shape=3,input_D=tools.input_D,input_H=tools.input_H,input_W=tools.input_W,phase='train')
train_loader = DataLoader(dataset=train_datasets, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset=val_datasets, batch_size=4, shuffle=True)
checkpoint = torch.load(tools.checkpoint_pretrain_resnet_10_23dataset,map_location=tools.device)
medicanet_resnet3d_10,parameters = generate_model(sample_input_W=tools.input_W,
                                                   sample_input_H=tools.input_H,
                                                   sample_input_D=tools.input_D,
                                                   num_seg_classes=tools.num_seg_classes,
                                                   phase='train',
                                                   pretrain_path=tools.checkpoint_pretrain_resnet_10_23dataset)
params = [
        { 'params': parameters['base_parameters'], 'lr': 0.001 },
        { 'params': parameters['new_parameters'], 'lr': 0.001*100 }
]
optimizer = optim.Adam(params, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
epochs = 200
print(medicanet_resnet3d_10)
tools.train_data(medicanet_resnet3d_10,train_loader,val_loader,epochs,optimizer,scheduler,tools.criterion,tools.outurl,tools.device)

