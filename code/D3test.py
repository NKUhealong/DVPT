import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import nibabel as nib
import cv2
import os
import time
import datetime
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch import optim
import math

class testDataSet(Dataset):
    def __init__(self, root, list_name, image_size):
        self.root = root
        self.list_path = root + list_name
        self.h, self.w = image_size
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        self.files = []
        for name in self.img_ids:
            image,GT = name.split(' ')
            img_file = os.path.join(self.root, "volume/%s" % image)
            label_file = os.path.join(self.root, "GT/%s" % GT)
            self.files.append({"img": img_file,"label": label_file})
        print("total {} samples".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        nii_image=nib.load(datafiles["img"]).get_fdata()
        gt_image=nib.load(datafiles["label"]).get_fdata() 
        _,_,slices = gt_image.shape
        image = np.zeros((self.h, self.w,slices))
        label = np.zeros((self.h, self.w,slices))
        
        for i in range(slices):
            image_slice = nii_image[:,:,i]
            label_slice = gt_image[:,:,i]
            image_slice = (image_slice-image_slice.min())/(image_slice.max() - image_slice.min())
            image_slice = cv2.resize(image_slice,(self.h, self.w),interpolation = cv2.INTER_NEAREST)
            label_slice = cv2.resize(label_slice,(self.h, self.w),interpolation = cv2.INTER_NEAREST)
            image[:,:,i] = image_slice*255
            label[:,:,i] = label_slice
        #print(np.unique(label))    
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        return image, label
        
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
############  for  ACDC ################
'''
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[2]):
        slice = image[:, :,ind]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            #print(pred.shape)
            result= np.zeros(shape=(patch_size[0], patch_size[0], 3), dtype=np.uint8)
            result[:, :, 0][np.where(pred == 1)] = 255
            result[:, :, 1][np.where(pred == 2)] = 255
            result[:, :, 2][np.where(pred == 3)] = 255
            cv2.imwrite(test_save_path+'ACDCPre'+str(ind)+'_h.png',result)
            
            cv2.imwrite(test_save_path+'ACDCimage'+str(ind)+'_h.png',image[:,:,ind])
            labelss= np.zeros(shape=(patch_size[0], patch_size[0], 3), dtype=np.uint8)
            labelss[:, :, 0][np.where(label[:,:,ind] == 1)] = 255
            labelss[:, :, 1][np.where(label[:,:,ind] == 2)] = 255
            labelss[:, :, 2][np.where(label[:,:,ind] == 3)] = 255
            cv2.imwrite(test_save_path+'ACDCGT'+str(ind)+'_h.png',labelss)
            
            prediction[:,:,ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

'''
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[2]):
        slice = image[:, :,ind]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        #input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            #print(pred.shape)
            
            result= np.zeros(shape=(patch_size[0], patch_size[0], 3), dtype=np.uint8)
            result[:, :][np.where(pred == 1)] = [255,0,255]
            result[:, :][np.where(pred == 2)] = [0,0,255]
            result[:, :][np.where(pred == 3)] = [0,255,0]
            result[:, :][np.where(pred == 4)] =[255,192,203]
            result[:, :][np.where(pred == 5)] =[255,255,0]
            result[:, :][np.where(pred == 6)] =[255,97,0]
            result[:, :][np.where(pred == 7)] =[128,42,42]
            result[:, :][np.where(pred == 8)] =[255,0,0]
            result[:, :][np.where(pred == 9)] =[218,112,214]
            result[:, :][np.where(pred == 10)] =[176,224,230]
            result[:, :][np.where(pred == 11)] =[56,94,15]
            result[:, :][np.where(pred == 12)] =[135,206,235]
            result[:, :][np.where(pred == 13)] =[227,207,87]
            '''
            masks = label[:, :,ind]
            GT= np.zeros(shape=(patch_size[0], patch_size[0], 3), dtype=np.uint8)
            GT[:, :][np.where(masks == 1)] = [255,0,255]
            GT[:, :][np.where(masks == 2)] = [0,0,255]
            GT[:, :][np.where(masks == 3)] = [0,255,0]
            GT[:, :][np.where(masks == 4)] =[255,192,203]
            GT[:, :][np.where(masks == 5)] =[255,255,0]
            GT[:, :][np.where(masks == 6)] =[255,97,0]
            GT[:, :][np.where(masks == 7)] =[128,42,42]
            GT[:, :][np.where(masks == 8)] =[255,0,0]
            GT[:, :][np.where(masks == 9)] =[218,112,214]
            GT[:, :][np.where(masks == 10)] =[176,224,230]
            GT[:, :][np.where(masks == 11)] =[56,94,15]
            GT[:, :][np.where(masks == 12)] =[135,206,235]
            GT[:, :][np.where(masks == 13)] =[227,207,87]
            '''
            cv2.imwrite(test_save_path+'pred'+str(ind)+'.jpg',result[:,:,::-1])
            #cv2.imwrite(test_save_path+'GT'+str(ind)+'.png',GT[:,:,::-1])
            #cv2.imwrite(test_save_path+'Synape'+str(ind)+'.png',image[:, :,ind])
            
            prediction[:,:,ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
   
    return metric_list


def inference(model, test_dir, image_size, num_classes,save_path):
    base_dir = test_dir
    image_size = image_size
    db_test = testDataSet(base_dir+'test/','test.txt', image_size)
    testloader = DataLoader(db_test, batch_size=1,shuffle=False, num_workers=4,pin_memory = True)
    model.eval()
    metric_list = 0.0
    for image,label in testloader:
        image, label = image.cuda(), label.cuda()
        metric_i = test_single_volume(image, label, model, classes=num_classes, patch_size=image_size,test_save_path=save_path, z_spacing=1)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    
    #RV_dice,RV_hd95 = metric_list[0][0],metric_list[0][1]
    #myo_dice,myo_hd95 = metric_list[1][0],metric_list[1][1]
    #LV_dice,LV_hd95 = metric_list[2][0],metric_list[2][1]
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    return mean_dice*100, mean_hd95
    
