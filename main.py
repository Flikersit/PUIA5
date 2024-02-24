import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import json
import matplotlib.image as mpimg 
from PIL import Image
from random import randint
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn
import sklearn.metrics
from sklearn.metrics import accuracy_score
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
        return self.traininputtensor[:, :, index], self.output[:, :, index]

    def __len__(self):
        return self.traininputtensor.size(dim=0)


class DataTest(Dataset):

    def __init__(self, data, annotation):
        self.testinputtensor = torch.tensor(data, dtype=torch.float)
        self.output = torch.tensor(annotation, dtype=torch.float)

    def __getitem__(self, index):
        return self.traininputtensor[:, :, index], self.output[:, :, index]

    def __len__(self):
        return self.traininputtensor.size(dim=0)


def changezax(data, z):
    while len(data) < z:
        data.append(np.zeros((512, 512), dtype=np.uint8))
    data = data[:z]
    return data



def splitToTest(annot, data, test=3):
    k = test
    print(data)
    traindata = data
    trainannot = annot
    testdata = []
    testannot = []
    data1 = []
    data2 = []
    while k!=0:
        rand = randint(0, len(traindata))
        print("Random", rand)
        print("Len", len(traindata))
        testdata.append(traindata[rand - 1])
        testannot.append(trainannot[rand - 1])
        traindata.pop(rand - 1)
        trainannot.pop(rand - 1)
        k -= 1
    for i in range(len(traindata)):
        data1.append({"image": traindata[i], "label": trainannot[i]})
    for k in range(len(testdata)):
        data2.append({"image": testdata[k], "label":testannot[k]})
    return[data1, data2]


import json
import numpy as np
from PIL import Image
from pathlib import Path



def preperingdata(path_to_masks, path_to_img, z):
    labels = []
    data = []
    for floader in path_to_masks.iterdir():
            print("Floader", floader)
            annot = []
            for file in floader.iterdir():
                img = Image.open(file)
                if img.mode != "L":
                    img = img.convert("L")
                img_array = np.array(img)
                annot.append(img_array)
            #print(np.array(changezax(annot, z=z)).shape)
            labels.append(changezax(annot, z=z))
    for floader1 in path_to_img.iterdir():
        for floader2 in floader1.iterdir():
            if floader2.is_dir():
                pic = []
                print(floader2)
                for picture in floader2.iterdir():
                    imga = Image.open(picture)
                    if imga.mode != "L":
                        imga = imga.convert("L")
                    array = np.array(imga)
                    pic.append(array)
                data.append(changezax(pic, z=z))
                #print(np.array(pic).shape)
    return [data, labels]
                    


    

# directory_path = Path(r'C:\Users\User\Desktop\pilsen_pigs_2023_cvat_backup\workspase')
# mask_path = Path(r'C:\Users\User\Desktop\pilsen_pigs_2023_cvat_backup\masks')
# data = preperingdata(mask_path, directory_path, z=800)


# data1 = splitToTest(data=data[0], annot=data[1])
# train_ds = CacheDataset(
#     data=data1[0],
#     cache_num=24,
#     cache_rate=1.0,
#     num_workers=8,
# )
# train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
# val_ds = CacheDataset(data=data1[1], cache_num=6, cache_rate=1.0, num_workers=4)
# val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
# num_epochs = 20
# rate_learning = 1e-4
# device = get_default_device()
# model = UNETR(
#     in_channels=1,
#     out_channels=10,
#     img_size=(512, 512, 800),
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)
# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# torch.backends.cudnn.benchmark = True
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(  # noqa: B038
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(r"D:\result", "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best



if __name__ == '__main__':
    directory_path = Path(r'C:\Users\User\Desktop\pilsen_pigs_2023_cvat_backup\workspase')
    mask_path = Path(r'C:\Users\User\Desktop\pilsen_pigs_2023_cvat_backup\masks')
    data = preperingdata(mask_path, directory_path, z=800)


    data1 = splitToTest(data=data[0], annot=data[1])
    train_ds = CacheDataset(
        data=data1[0],
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    val_ds = CacheDataset(data=data1[1], cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    num_epochs = 20
    rate_learning = 1e-4
    device = get_default_device()
    model = UNETR(
        in_channels=1,
        out_channels=10,
        img_size=(512, 512, 800),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    max_iterations = 100
    eval_num = 500
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
    model.load_state_dict(torch.load(os.path.join(r"D:\result", "best_metric_model.pth")))




    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.show()





                        
