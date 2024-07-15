import numpy as np
import nibabel as nib
from pathlib import Path
import json 
from PIL import Image
from random import randint
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import torch.nn.functional as F
import copy
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from tqdm import tqdm
import os
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import pickle



def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataloader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

class DataTrain(Dataset):

    def __init__(self, data, annotation):
        self.traininputtensor = torch.tensor(data, dtype=torch.float)
        self.output = torch.tensor(annotation, dtype=torch.float)
    
    def __getitem__(self, index):
        input_image = self.traininputtensor[index].unsqueeze(0)  
        output_label = self.output[index].unsqueeze(0)  
        return input_image, output_label

    def __len__(self):
        return self.traininputtensor.size(dim=0)


class DataTest(Dataset):

    def __init__(self, data, annotation):
        self.testinputtensor = torch.tensor(data, dtype=torch.float)
        self.output = torch.tensor(annotation, dtype=torch.float)

    def __getitem__(self, index):
        input_image = self.testinputtensor[index].unsqueeze(0) 
        output_label = self.output[index].unsqueeze(0) 
        return input_image, output_label

    def __len__(self):
        return self.testinputtensor.size(dim=0)


def changezax(data, z):
    while len(data) < z:
        data.append(np.zeros((96, 96), dtype=np.uint8))
    data = data[:z]
    return data


def splitToTest1(annot, data, test=3):
    k = test
    traindata = data.copy()
    trainannot = annot.copy()
    testdata = []
    testannot = []
    testdatafinal = []
    testannotfinal = []
    trainannotfinal = []
    traindatafinal = []
    while k!=0:
        rand = randint(0, len(traindata)-1)
        testdata.append(traindata[rand])
        testannot.append(trainannot[rand])
        traindata.pop(rand - 1)
        trainannot.pop(rand - 1)
        k -= 1
    for i in range(len(testdata)):
        testdatafinal.append(np.array(testdata[i]))
        testannotfinal.append(np.array(testannot[i]))
    for k in range(len(traindata)):
        traindatafinal.append(np.array(traindata[k]))
        trainannotfinal.append(np.array(trainannot[k]))
    
    return[testdatafinal, testannotfinal, traindatafinal, trainannotfinal]




def preperingdata(path_to_masks, path_to_img, z):
    labels = []
    data = []
    for floader in path_to_masks.iterdir():
            print("Floader", floader)
            annot = []
            for file in floader.iterdir():
                img = Image.open(file)
                if img.mode != "1":
                    img = img.convert("1")
                img = img.resize((96, 96))
                img_array = np.array(img)
                annot.append(img_array)
            labels.append(changezax(annot, z=z))
    for floader1 in path_to_img.iterdir():
        if floader1.is_dir():
            pic = []
            print("Floader", floader1)
            for picture in floader1.iterdir():
                imga = Image.open(picture)
                if imga.mode != "L":
                    imga = imga.convert("L")
                imga = imga.resize((96, 96))
                array = np.array(imga)
                pic.append(array)
            data.append(changezax(pic, z=z))

    return [data, labels]





directory_path = Path(r'/storage/brno2/home/yauheni/cuted/data')
mask_path = Path(r'/storage/brno2/home/yauheni/cuted/masks')
data = preperingdata(mask_path, directory_path, z=272)





device = get_default_device()
model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(272, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)
modelwise = UNETR(   #just_to_copy_weights
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)
modelwise.load_state_dict(torch.load(os.path.join(r"/storage/brno2/home/yauheni", "best_metric_model.pth")))
model_state_dict1 = modelwise.state_dict()
for name_dst, param_dst in model.named_parameters():
    if name_dst in modelwise.state_dict():
        param_src = model.state_dict()[name_dst]
        if param_src.size() == param_dst.size():
            param_dst.data.copy_(param_src.data)
        else:
            print(f"Skipping layer {name_dst} due to size mismatch")










root_dir = r'/storage/brno2/home/yauheni'
data2 = splitToTest1(data=data[0], annot=data[1])
train = DataTrain(data2[2], data2[3])
test = DataTest(data2[0], data2[1])
dataloader_train = DataLoader(dataset=train, batch_size=1, shuffle=True)
dataloader_test = DataLoader(dataset=test, batch_size=1, shuffle=True)
dice_metric_with_bg = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)



loss_fn =  DiceCELoss(to_onehot_y=True, sigmoid=True)
lr = 1e-4
num_epochs = 2500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimazer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
history = []
history1 = []
acc_val = []
acc_test = []
acc_train = []
dice_metric_best = 0
number_of_epoch_best = 0
for epochs in range(num_epochs):
    with tqdm(total=17, desc=f'Epoch {epochs + 1}/{num_epochs}', unit='batch') as pbar:
        running_loss = 0
        val_loss = 0

        model.train()
        for i, data in enumerate(dataloader_train):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimazer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimazer.step()
            running_loss += loss.item()


            preds = (labels>0.5).float()
            output = (output>0.5).float()
            dice_metric_with_bg(y_pred=output, y=preds)
            dice_score = dice_metric_with_bg.aggregate().item()
            acc_train.append(dice_score)
            dice_metric_with_bg.reset()
            pbar.update(1)
        history.append(running_loss)
    correct = 0
    total = 0
    for j, data in enumerate(dataloader_test):
        with torch.no_grad():
            inputs_for_test, labels_for_test = data
            inputs_for_test = inputs_for_test.to(device)
            labels_for_test = labels_for_test.to(device)
            output_for_test = model(inputs_for_test)
            preds = (labels_for_test>0.5).float()
            output = (output_for_test>0.5).float()
            dice_metric_with_bg(y_pred=output, y=preds)
    dice_score = dice_metric_with_bg.aggregate().item()
    acc_val.append(dice_score)
    if dice_score>dice_metric_best:
        number_of_epoch_best = epochs
        dice_metric_best = dice_score
        torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model_pigs.pth"))
        print(
                "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_metric_best, dice_score)
        )
    else:
        print(
                "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_metric_best, dice_score
                    )
                )
        dice_metric_with_bg.reset()
print("Train history", history)
print("Accuracy train", acc_train)
print("Accuracy validation", acc_val)





file_path = r'/storage/brno2/home/yauheni/history_pigs.pkl'


if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(history, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(history, file)
    print(f"Data saved to existing file: {file_path}")


file_path = r'/storage/brno2/home/yauheni/acc_train_pigs.pkl'


if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(acc_train, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(acc_train, file)
    print(f"Data saved to existing file: {file_path}")



file_path = r'/storage/brno2/home/yauheni/acc_val.pkl'


if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(acc_val, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(acc_val, file)
    print(f"Data saved to existing file: {file_path}")

print("The end")
