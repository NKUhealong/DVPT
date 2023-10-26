from PIL import ImageFilter
import random
from PIL import Image
import math
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
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        #print(input_tensor.size())
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs.size(),target.size())
        
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[0]-1
    posemb_tok, posemb_grid =  posemb[0:1,:], posemb[1:, :]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=0)
    return posemb

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def adjust_learning_rate(args, optimizer, epoch):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.learning_rate * epoch / args.warmup_epochs
    else:
        lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
 

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def initialize_optimizer(args, model):
    optimizer_strategy = args.optimizer
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    momentum = args.momentum
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
    elif optimizer_strategy == 'ADAM':
         
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif optimizer_strategy == 'ADAMW':
        optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    else:
        raise NotImplementedError('Not implemented optimizer.')
    return optimizer

class Estimator():
    def __init__(self, criterion, num_classes, thresholds=None):
        self.criterion = criterion
        self.num_classes = num_classes
        self.thresholds = [-0.5 + i for i in range(num_classes)] if not thresholds else thresholds
        self.reset()  # intitialization

    def update(self, predictions, targets):
        targets = targets.cpu()
        predictions = predictions.cpu()
        predictions = self.to_prediction(predictions)

        # update metrics
        self.num_samples += len(predictions)
        self.correct += (predictions == targets).sum().item()
        for i, p in enumerate(predictions):
            self.conf_mat[int(targets[i])][int(p.item())] += 1

    def get_accuracy(self, digits=-1):
        acc = self.correct / self.num_samples
        acc = acc if digits == -1 else round(acc, digits)
        return acc

    def get_kappa(self, digits=-1):
        kappa = quadratic_weighted_kappa(self.conf_mat)
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa

    def reset(self):
        self.correct = 0
        self.num_samples = 0
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def to_prediction(self, predictions):
        if self.criterion == 'ce':
            predictions = torch.tensor([torch.argmax(p) for p in predictions]).long()
        elif self.criterion == 'mse':
            predictions = torch.tensor([self.classify(p.item()) for p in predictions]).float()
        else:
            raise NotImplementedError('Not implemented criterion.')

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i

class Evaluator:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.Dice = list()       
        self.IoU_polyp = list()

    def evaluate(self, pred, gt):
        
        pred_binary = (pred >= 0.5).float().cuda()
        pred_binary_inverse = (pred_binary == 0).float().cuda()

        gt_binary = (gt >= 0.5).float().cuda()
        gt_binary_inverse = (gt_binary == 0).float().cuda()
        
        TP = pred_binary.mul(gt_binary).sum().cuda(0)
        FP = pred_binary.mul(gt_binary_inverse).sum().cuda()
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda()
        FN = pred_binary_inverse.mul(gt_binary).sum().cuda()

        if TP.item() == 0:
            TP = torch.Tensor([1]).cuda()
            
        Recall = TP / (TP + FN)
        # Precision or positive predictive value
        Precision = TP / (TP + FP)
        # F1 score = Dice
        Dice = 2 * Precision * Recall / (Precision + Recall)
        # Overall accuracy
        # IoU for poly
        IoU_polyp = TP / (TP + FP + FN)
        return  Dice.data.cpu().numpy().squeeze(), IoU_polyp.data.cpu().numpy().squeeze()

        
    def update(self, pred, gt):
        dice, ioU_polyp = self.evaluate(pred, gt)        
        self.Dice.append(dice)       
        self.IoU_polyp.append(ioU_polyp)

    def show(self,flag = True):
        if flag == True:
            print(" Dice: ", "%.2f" % (np.mean(self.Dice)*100)," IoU: ", "%.2f" % (np.mean(self.IoU_polyp)*100),'\n')
        
        return np.mean(self.Dice)*100,np.mean(self.IoU_polyp)*100


def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)
            
def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    #print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    #print('=' * max_len)


def print_config(args):
    print('=================================')
    for key, value in args.__dict__.items():
        print('{}: {}'.format(key, value))
    print('=================================')


def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets
    print('=========================')
    print('Dataset Loaded.')
    #print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    print('=========================')