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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from models import *
from utils import *
from dataset import *
from segFormer import *
from Our_segformer import *
#from D3test import *
#from Dataset3D import *

parser = argparse.ArgumentParser()

parser.add_argument('--data-path',    type=str, default='./data/polyp/', help='idrid  polyp, skin   drivefull')
parser.add_argument('--dataset',       type=str, default= 'polyp', help='idrid  polyp  skin  drive ACDC synape')
parser.add_argument('--save-path',    type=str, default='./new/', help='save_path')
parser.add_argument('--model-name',   type=str, default='swin_base 3 channel', help='RN50x4, ViTB16')
parser.add_argument('--optimizer',    type=str, default='ADAMW', help='optimizer')
parser.add_argument('--pretain',      type=int, default=True, help='log path')
parser.add_argument('--pre',          type=str, default='our_Seg_Mode', help='model architecture')
parser.add_argument('--input-size',   type=int, default=512, help='input size') 
parser.add_argument('--batch-size',   type=int, default=10, help='batch size')


parser.add_argument('--epochs',       type=int, default=80, help='number of epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--learning-rate',type=float, default=0.00002, help='learning rate')
parser.add_argument('--momentum',     type=float, default=0.9, help='learning rate')
parser.add_argument('--num-classes',  type=int, default=2, help='number of classes')
parser.add_argument('--device',       type=str, default='cuda', help='device')
parser.add_argument('--criterion',    type=str, default='ce', help='mse / ce')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--disable-progress', default=True,  help='disable progress bar')


def main(args):
    print_config(args)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    set_random_seed(2023)
    #model = EVP_SegFormer_B4(args, num_classes = 2)
    #model = Swin_Seg_Model(args, num_classes = 2)
    #model = SegFormer_B4(args.input_size,2)
    model = our_SegFormer_B4(args.input_size,2)
    #model = SSF_SegFormer_B4(args.input_size,2) 
    #model = Adapter_SegFormer_B4(args.input_size,2)
    model = model.to(args.device)
    train_loader, val_loader, test_loader = generate_dataset(args)
    for name,p in model.named_parameters():
        if 'linear_pred'  in name: 
            p.requires_grad = True
        elif 'prompt' in name or 'ssf' in name or 'adaptmlp' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    '''        
    x=torch.rand((1,3,512,512)).to('cuda')
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
    flops = FlopCountAnalysis(model, x)
    acts = ActivationCountAnalysis(model, x)
    print(f"total flops : {(flops.total()+acts.total())/1e9}",'G')
    '''
    print('Total model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
    print('Trainable model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad == True)/1e6,'M' )
    scaler = torch.cuda.amp.GradScaler()
    train(args, model, train_loader, val_loader, scaler=scaler )
 
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, args.model_name+args.pre+'_final_seg_weights.pt')
    evaluate(args, model, checkpoint, test_loader, test = False)
    
    print('This is the performance of the best validation model')
    checkpoint = os.path.join(save_path, args.model_name+args.pre+'_best_validation_seg_weights.pt')
    evaluate(args, model, checkpoint, test_loader, test = True)
    

def train(args, model, train_loader, val_loader, scaler=None):
    device = args.device
    optimizer = initialize_optimizer(args, model)
    ce_loss = nn.CrossEntropyLoss() 
    dice_loss = DiceLoss(args.num_classes)

    model.float().train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress = tqdm(enumerate(train_loader)) if not args.disable_progress else enumerate(train_loader)
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device)
            
            with torch.cuda.amp.autocast():
                outputs= model(X)
                #print(outputs.shape)
                outputs_soft = torch.softmax(outputs, dim=1)
                loss = 0.5*ce_loss(outputs,y.long())+0.5*dice_loss(outputs_soft, y.long())
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            #nn.utils.clip_grad_norm_(model.parameters(), 1)

            scaler.step(optimizer)
            scaler.update()
            '''
            y_pred = model(X)
             
            loss = loss_function(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            '''

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
                         
            curr_lr = optimizer.param_groups[0]['lr']
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            message = '[{}] epoch: [{} / {}], loss: {:.6f}, LR: {:.6f}'.format(current_time, epoch + 1, args.epochs, avg_loss, curr_lr)
            if not args.disable_progress:
                progress.set_description(message)

        if args.disable_progress:
            print(message)

        if epoch % 1 == 0:
            Dice, IoU = eval(model, val_loader, device)
            print('Validation Dice: {}, IoU: {}'.format(Dice, IoU))
            
            indicator = Dice+IoU  
            if indicator > max_indicator:
                torch.save(model.state_dict(), os.path.join(args.save_path, args.model_name+args.pre+'_best_validation_seg_weights.pt'))
                max_indicator = indicator

        adjust_learning_rate(args, optimizer, epoch)
    torch.save(model.state_dict(), os.path.join(args.save_path, args.model_name+args.pre+'_final_seg_weights.pt'))
    
def evaluate(args, model, checkpoint, test_loader, test = False):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)  
    Dice, IoU = eval(model, test_loader, args.device)

    print('========================================')
    print('Finished! test Dice: {}'.format(round(Dice, 2)),'   IoU: {}'.format(round(IoU, 2)))
    if test:
        with open("./eval_seg.txt","a") as f:
            txt = 'test Dice:'+str(round(Dice, 2))+' IoU:'+str(round(IoU, 2))+' LR = '+str(args.learning_rate)+args.model_name+args.dataset
            f.write(txt+'\n') 
    print('========================================')

def eval(model, dataloader, device):
    model.eval()
    torch.set_grad_enabled(False)
    save_dir= './result/'
    j = 0

    evaluator = Evaluator()
    for test_data in dataloader:
        images, labels = test_data['image'].to(device), test_data['label'].to(device)

        predictions = model(images)
        pred = predictions[0,1,:,:]
        evaluator.update(pred, labels[0,:,:].float())
        for i in range(1):
            #images = images[i].cpu().numpy()
            #labels = labels.cpu().numpy()
            #label = (labels[i]*255)
            pred = pred.cpu().numpy()
            #cv2.imwrite(save_dir+'image'+str(j)+'.jpg',images.transpose(1, 2, 0)[:,:,::-1])
            #cv2.imwrite(save_dir+'GT'+str(j)+'.png',label*255)
            cv2.imwrite(save_dir+'Pre'+str(j)+'.png',pred*255)
            j=j+1

    Dice, IoU = evaluator.show(False)

    model.train()
    torch.set_grad_enabled(True)
    return Dice, IoU

def generate_dataset(args):
    db_train = MyDataSet(args.data_path+'train/', 'train.txt',360,(args.input_size,args.input_size),args.dataset)
    train_loader = DataLoader(db_train, batch_size=args.batch_size,num_workers=10, pin_memory=True,drop_last=False)

    db_val = testBaseDataSets(args.data_path+'test/', 'test.txt',(args.input_size,args.input_size),args.dataset)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=4)
  
    dataset = db_train, db_val, db_val

    print_dataset_info(dataset)
    return train_loader,valloader,valloader

if __name__ == '__main__':
     
    
    lr = 0.0005
    args = parser.parse_args()
    args.learning_rate = lr
    main(args)
            
    