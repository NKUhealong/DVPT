import os
import time
import random
import argparse

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models import *
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--data-path',    type=str, default='./data/DDRCls/', help='dataset folder')
parser.add_argument('--save-path',    type=str, default='./new/', help='save_path')
parser.add_argument('--model-name',   type=str, default='ViTB16', help='RN50x4, ViTB16')
parser.add_argument('--optimizer',    type=str, default='ADAMW', help='optimizer')
parser.add_argument('--pretain',      type=int, default=True, help='log path')
parser.add_argument('--pre',          type=str, default='ViT_CLS_prompt_Models', help='model architecture')
parser.add_argument('--input-size',   type=int, default=288, help='input size') 
parser.add_argument('--batch-size',   type=int, default=50, help='batch size')


parser.add_argument('--epochs',       type=int, default=60, help='number of epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--learning-rate',type=float, default=0.00002, help='learning rate')
parser.add_argument('--momentum',     type=float, default=0.9, help='learning rate')
parser.add_argument('--dataset',      type=str, default='ddr', help='ddr / aptos2019 / messidor2')
parser.add_argument('--num-classes',  type=int, default=5, help='number of classes')
parser.add_argument('--device',       type=str, default='cuda', help='device')
parser.add_argument('--criterion',    type=str, default='ce', help='mse / ce')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--disable-progress', default=True,  help='disable progress bar')


def main(args):
    print_config(args)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    set_random_seed(2023)
    model = ViT_Model(args, num_classes=5)  #  ViT_AdptFormer_Model   ViT_Model  ViT_VPT_Model ViT_EVP_Model  ViT_SSF_Model
    model = model.to(args.device)
    #print(model.state_dict().keys())
    for name,p in model.named_parameters():
        if 'fc_head'  in name:   # adaptmlp
            p.requires_grad = True
        elif 'prompt'  in name or 'ssf'  in name or 'adaptmlp'  in name:  #or 'bias'  in name   embedding lightweight_mlp
            p.requires_grad = True
        else:
            p.requires_grad = False
    
    print('Total model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
    print('Trainable model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad == True)/1e6,'M' )
    train_dataset, test_dataset, val_dataset = generate_dataset(args)
    estimator = Estimator(args.criterion, args.num_classes)
    scaler = torch.cuda.amp.GradScaler()
    train(args=args, model=model, train_dataset=train_dataset, val_dataset=val_dataset, estimator=estimator, scaler=scaler )

   
    print('This is the performance of the best validation model')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    checkpoint = os.path.join(save_path, args.model_name+args.pre+'_best_validation_weights.pt')
    evaluate(args, model, checkpoint, test_dataset, estimator, test = True)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, args.model_name+args.pre+'_final_weights.pt')
    evaluate(args, model, checkpoint, test_dataset, estimator, test = False)


def train(args, model, train_dataset, val_dataset, estimator, scaler=None):
    device = args.device
    optimizer = initialize_optimizer(args, model)
    loss_function = nn.CrossEntropyLoss()    
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,num_workers=8,
                              drop_last=True,pin_memory=True, sampler=sampler_train)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=8, pin_memory=True)

    model.float().train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader)) if not args.disable_progress else enumerate(train_loader)
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device).long()
            
            with torch.cuda.amp.autocast():
                y_pred= model(X)
                loss = loss_function(y_pred, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            scaler.step(optimizer)
            scaler.update()
    
            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)
            avg_acc = estimator.get_accuracy(4)
            avg_kappa = estimator.get_kappa(4)
            curr_lr = optimizer.param_groups[0]['lr']
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            message = '[{}] epoch: [{} / {}], loss: {:.6f}, acc: {:.4f}, kappa: {:.4f}, LR: {:.6f}'.format(current_time, epoch + 1, args.epochs, avg_loss, avg_acc, avg_kappa, curr_lr)
            if not args.disable_progress:
                progress.set_description(message)

        if args.disable_progress:
            print(message)

        if epoch % 1 == 0:
            eval(model, val_loader, estimator, device)
            acc = estimator.get_accuracy(4)
            kappa = estimator.get_kappa(4)
            print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
            indicator = kappa 
            if indicator > max_indicator:
                torch.save(model.state_dict(), os.path.join(args.save_path, args.model_name+args.pre+'_best_validation_weights.pt'))
                max_indicator = indicator

        adjust_learning_rate(args, optimizer, epoch)
    torch.save(model.state_dict(), os.path.join(args.save_path, args.model_name+args.pre+'_final_weights.pt'))
    
def evaluate(args, model, checkpoint, test_dataset, estimator, test = False):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=8,shuffle=False,pin_memory=True)

    eval(model, test_loader, estimator, args.device)

    print('========================================')
    print('Finished! test acc: {}'.format(estimator.get_accuracy(4)),'quadratic kappa: {}'.format(estimator.get_kappa(4)))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
    if test:
        with open("./eval.txt","a") as f:
            txt = 'test acc:'+str(estimator.get_accuracy(4))+' quadratic kappa:'+str(estimator.get_kappa(4))+' LR = '+str(args.learning_rate)+args.pre
            f.write(txt+'\n') 
    
    print('========================================')

def eval(model, dataloader, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        X, y = test_data
        X, y = X.to(device), y.to(device).float()
        y_pred= model(X)
        estimator.update(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)

def generate_dataset(args):
    train_transform, test_transform = data_transforms(args)
    train_path = os.path.join(args.data_path, 'train')
    test_path = os.path.join(args.data_path, 'test')
    val_path = os.path.join(args.data_path, 'valid')

    train_dataset = datasets.ImageFolder(train_path, train_transform, loader=pil_loader)
    ratio = 1.0
    s1,s2,s3,s4,s5 = int(2992*ratio),int(1638*ratio),int(2238*ratio),int(613*ratio),int(2370*ratio)
    train_dataset.samples = train_dataset.samples[0:s1]+train_dataset.samples[2992:2992+s2]+train_dataset.samples[2992+1638:2992+1638+s3]+train_dataset.samples[2992+1638+2238:2992+1638+2238+s4]+train_dataset.samples[2992+1638+2238+613:2992+1638+2238+613+s5]
    test_dataset = datasets.ImageFolder(test_path, test_transform, loader=pil_loader)
    val_dataset = datasets.ImageFolder(val_path, test_transform, loader=pil_loader)

    dataset = train_dataset, test_dataset, val_dataset
    print_dataset_info(dataset)
    return dataset

def data_transforms(args):
    augmentations = [transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),
                     transforms.RandomResizedCrop(size=(args.input_size, args.input_size),scale=(0.87, 1.15),ratio=(0.7, 1.3)),
                     transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1,hue=0.1),
                     transforms.RandomRotation(degrees=(-180, 180)),transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                     transforms.Resize((args.input_size, args.input_size)),
                     transforms.ToTensor()]

    train_preprocess = transforms.Compose(augmentations)

    test_preprocess = transforms.Compose([transforms.Resize((args.input_size, args.input_size)),transforms.ToTensor()])
    return train_preprocess, test_preprocess

if __name__ == '__main__':
    lr = 0.0005
    args = parser.parse_args()
    args.learning_rate = lr
    main(args)
        