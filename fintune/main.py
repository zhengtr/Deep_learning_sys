# -*- coding: utf-8 -*-
"""SwAVTest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y_Y4Jn3CAcaZIBfjoh0C6DEVxE5Vqv62
"""

# pip install fastai==2.4

import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import argparse
from str2bool import str2bool
from load_swav import load_pretrained_swav

parser = argparse.ArgumentParser(description='Transfer Resnet50 to Unet')
parser.add_argument("--saveImage", type=str2bool, default=False)
parser.add_argument("--train", type=str2bool, default=True)
parser.add_argument("--modelPath", type=str, default="default.pt")
# parser.add_argument("--saveModel", type=str2bool, default=True)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--visualNum", type=int, default=5)
parser.add_argument("--visualFreq", type=int, default=1)
parser.add_argument("--givenModel", type=str, default="")
args = parser.parse_args()

Path.ls = lambda x: list(x.iterdir())

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_colab = None

from fastai.data.external import untar_data, URLs

coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
use_colab = True

if use_colab == True:
    path = coco_path
else:
    path = "Your path to the dataset"

paths = glob.glob(path + "/*.jpg")  # Grabbing all the image file names
np.random.seed(123)
paths_subset = np.random.choice(paths, 10_000, replace=False)  # choosing 1000 images randomly
rand_idxs = np.random.permutation(10_000)
train_idxs = rand_idxs[:8000]  # choosing the first 8000 as training set
val_idxs = rand_idxs[8000:]  # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")

"""## Dataset DataLoader"""

SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader


train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

"""## Loss"""


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def visualize_eval(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
        fake_color = model.fake_color.detach()
        real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)
        real_imgs = lab_to_rgb(L, real_color)
        fig = plt.figure(figsize=(15, 8))
        for i in range(5):
            ax = plt.subplot(3, 5, i + 1)
            ax.imshow(L[i][0].cpu(), cmap='gray')
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 5)
            ax.imshow(fake_imgs[i])
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 10)
            ax.imshow(real_imgs[i])
            ax.axis("off")
        plt.show()
        if save:
            fig.savefig(f"colorization_{time.time()}.png")

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


"""## Use SwAV"""

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet50
from fastai.vision.models.unet import DynamicUnet
from torch import nn, optim


def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model50 = load_pretrained_swav("/content/drive/My Drive/swav_ckp_190.pth")
    body = nn.Sequential(*list(model50.children())[:-2])
    net_G = DynamicUnet(body, 2, (256, 256)).to(device)
    net_G = net_G.to(device)
    # model50 = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    # body = nn.Sequential(*list(model50.children())[:-2])
    # body = create_body(resnet50, pretrained=True, n_in=n_input, cut=-2)
    # net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G


def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in train_dl:
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), L.size(0))
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")


class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

    def setup_RGB(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        self.O = data['O'].to(self.device)

    def forward_RGB(self):
        self.my_fake_color = self.net_G(self.O)

def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

# Patch Discriminiator
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False,
                                  act=False)]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True,
                   act=True):  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)]  # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# GAN Loss
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
# Train Model
def train_model(model, val_dl, epochs, display_every=200, save=False):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network
        for data in train_dl:
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=save) # function displaying the model's outputs


class ResUNetDataSet(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        # imgO = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        temp = L.clone().detach()
        imgO = torch.cat((temp, temp, temp), 0)
        return {'L': L, 'ab': ab, 'O': imgO}

    def __len__(self):
        return len(self.paths)


def make_ResUNet_dataloaders(batch_size=2, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    dataset = ResUNetDataSet(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    return dataloader


def pretrain_ResUNet_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in train_dl:
            L, ab, O = data['L'].to(device), data['ab'].to(device), data['O'].to(device)
            preds = net_G(O)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

def train_resUNet(epochs=50, save=True):
    # model50 = load_pretrained_swav(args.modelPath)
    # model50 = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    # body = nn.Sequential(*list(model50.children())[:-2])
    # net_G = DynamicUnet(body, 2, (256, 256)).to(device)
    #If you want to use your own model uncommon then following
    #net_G.load_state_dict(torch.load(args.modelPath, map_location=device))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model50 = load_pretrained_swav(args.givenModel)
    print(model50)
    print("=====================================")
    body = nn.Sequential(*list(model50.children())[:-2])
    net_G = DynamicUnet(body, 2, (256, 256)).to(device)
    net_G = net_G.to(device)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    train_loader = make_ResUNet_dataloaders(paths=train_paths, split='train')
    # val_loader = make_ResUNet_dataloaders(paths=train_paths, split='val')
    pretrain_ResUNet_generator(net_G, train_loader, opt, criterion, epochs)
    if save:
        torch.save(net_G.state_dict(), args.modelPath)

def eval_model(model, save=True):
    val_loader = make_ResUNet_dataloaders(paths=train_paths, split='val')
    n = 0
    q = 0
    for data in val_loader:
        if n > args.visualNum:
            break
        if q % args.visualFreq == 0:
            with torch.no_grad():
                model.setup_RGB(data)
                model.forward_RGB()
            fake_color = model.my_fake_color.detach()
            real_color = model.ab
            L = model.L
            fake_imgs = lab_to_rgb(L, fake_color)
            real_imgs = lab_to_rgb(L, real_color)
            fig = plt.figure(figsize=(15, 8))
            for i in range(2):
                ax = plt.subplot(3, 5, i + 1)
                ax.imshow(L[i][0].cpu(), cmap='gray')
                ax.axis("off")
                ax = plt.subplot(3, 5, i + 1 + 5)
                ax.imshow(fake_imgs[i])
                ax.axis("off")
                ax = plt.subplot(3, 5, i + 1 + 10)
                ax.imshow(real_imgs[i])
                ax.axis("off")
            plt.show()
            n+=1
            if save:
                fig.savefig(f"colorization_{time.time()}.png")
        q+=1

# def main():

#     net_G = build_res_unet(n_input=1, n_output=2, size=256)
#     opt = optim.Adam(net_G.parameters(), lr=1e-4)
#     criterion = nn.L1Loss()
#     pretrain_generator(net_G, train_dl, opt, criterion, 50)
#     if args.saveModel:
#         torch.save(net_G.state_dict(), args.modelPath)

def load_ResUNet():
    model50 = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    body = nn.Sequential(*list(model50.children())[:-2])
    net_G = DynamicUnet(body, 2, (256, 256)).to(device)
    net_G.load_state_dict(torch.load(args.modelPath, map_location=device))
    return net_G

if __name__ == "__main__":
    if args.train:
        print("start Train")
        train_resUNet(epochs=args.epochs, save=args.saveImage)
    else:
        print("start Eval")
        model50 = load_pretrained_swav("swav_ckp_190.pth")
        body = nn.Sequential(*list(model50.children())[:-2])
        uNet = DynamicUnet(body, 2, (256, 256)).to(device)
        uNet.load_state_dict(torch.load(args.givenModel, map_location=device))
        uNet = uNet.to(device)
        model = MainModel(net_G=uNet)
        eval_model(model, args.saveImage)