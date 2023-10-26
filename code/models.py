import os
import time
import random
import argparse
import clip
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import *
from adptformer import *


class ViT_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(ViT_Model, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'RN50x4' if args.model_name =='RN50x4' else  'ViT-B/16'
        model, preprocess = clip.load(args, model_name, device)
        model_dict = model.state_dict()
        if args.model_name !='RN50x4':
            checkpoint = torch.load('./ViT-B_16.pt', map_location="cpu")
            new_checkpoint = {}
            for k in list(checkpoint.keys()):
                if k.startswith('visual.'):
                    new_checkpoint[k] = checkpoint[k]
            del checkpoint
            pos_embed_w = resize_pos_embed(new_checkpoint['visual.positional_embedding'],model.state_dict()['visual.positional_embedding'], num_tokens=1, gs_new=(18,18))# 18,18
            new_checkpoint['visual.positional_embedding'] = pos_embed_w
            matched_dict = {k: v for k, v in new_checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
            print('matched keys:', len(matched_dict))
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
        else:
            model_dict = model.state_dict()
            checkpoint = torch.load('./RN50x4.pt', map_location="cpu")
            new_checkpoint = {}
            for k in list(checkpoint.keys()):
                if k.startswith('visual.'):
                    new_checkpoint[k] = checkpoint[k]
            del checkpoint
            matched_dict = {k: v for k, v in new_checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
            print('matched keys:', len(matched_dict))
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            
        model.transformer = None
        model.token_embedding = None
        model.ln_final = None
        model.text_projection = None
        model.logit_scale = None
        self.encoder = model.visual
        embed = 640 if args.model_name =='RN50x4' else 512
        self.fc_head = nn.Linear(embed, num_classes, bias=False)
        
    def forward(self, x):
        image_features = self.encoder(x)
        logits = self.fc_head(image_features)
        return logits
        
class ViT_SSF_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(ViT_SSF_Model, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'RN50x4' if args.model_name =='RN50x4' else  'ViT-B/16'
        model = ViT_SSF(args, model_name, device)
        model_dict = model.state_dict()
        if args.model_name !='RN50x4':
            checkpoint = torch.load('./ViT-B_16.pt', map_location="cpu")
            new_checkpoint = {}
            for k in list(checkpoint.keys()):
                if k.startswith('visual.'):
                    new_checkpoint[k] = checkpoint[k]
            del checkpoint
            pos_embed_w = resize_pos_embed(new_checkpoint['visual.positional_embedding'],model.state_dict()['visual.positional_embedding'], num_tokens=1, gs_new=(18,18))
            new_checkpoint['visual.positional_embedding'] = pos_embed_w
            matched_dict = {k: v for k, v in new_checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
            print('matched keys:', len(matched_dict))
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            
        model.transformer = None
        model.token_embedding = None
        model.ln_final = None
        model.text_projection = None
        model.logit_scale = None
        self.encoder = model.visual
        embed = 640 if args.model_name =='RN50x4' else 512
        self.fc_head = nn.Linear(embed, num_classes, bias=False)
    def forward(self, x):
        
        image_features = self.encoder(x)
        #print(image_features.shape)
        logits = self.fc_head(image_features)
        return logits

class ViT_AdptFormer_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(ViT_AdptFormer_Model, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'RN50x4' if args.model_name =='RN50x4' else  'ViT-B/16'
        model = load_AdaptFormer(args, model_name, device)
        model_dict = model.state_dict()
        if args.model_name !='RN50x4':
            checkpoint = torch.load('./ViT-B_16.pt', map_location="cpu")
            new_checkpoint = {}
            for k in list(checkpoint.keys()):
                if k.startswith('visual.'):
                    new_checkpoint[k] = checkpoint[k]
            del checkpoint
            pos_embed_w = resize_pos_embed(new_checkpoint['visual.positional_embedding'],model.state_dict()['visual.positional_embedding'], num_tokens=1, gs_new=(18,18))
            new_checkpoint['visual.positional_embedding'] = pos_embed_w
            matched_dict = {k: v for k, v in new_checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
            print('matched keys:', len(matched_dict))
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            
        model.transformer = None
        model.token_embedding = None
        model.ln_final = None
        model.text_projection = None
        model.logit_scale = None
        self.encoder = model.visual
        embed = 640 if args.model_name =='RN50x4' else 512
        self.fc_head = nn.Linear(embed, num_classes, bias=False)
    def forward(self, x):
        
        image_features = self.encoder(x)
        #print(image_features.shape)
        logits = self.fc_head(image_features)
        return logits
        
class ViT_EVP_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(ViT_EVP_Model, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'RN50x4' if args.model_name =='RN50x4' else  'ViT-B/16'
        model = ViT_EVP(args, model_name, device)
        model_dict = model.state_dict()
        if args.model_name !='RN50x4':
            checkpoint = torch.load('./ViT-B_16.pt', map_location="cpu")
            new_checkpoint = {}
            for k in list(checkpoint.keys()):
                if k.startswith('visual.'):
                    new_checkpoint[k] = checkpoint[k]
            del checkpoint
            pos_embed_w = resize_pos_embed(new_checkpoint['visual.positional_embedding'],model.state_dict()['visual.positional_embedding'], num_tokens=1, gs_new=(18,18))
            new_checkpoint['visual.positional_embedding'] = pos_embed_w
            matched_dict = {k: v for k, v in new_checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
            print('matched keys:', len(matched_dict))
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            
        model.transformer = None
        model.token_embedding = None
        model.ln_final = None
        model.text_projection = None
        model.logit_scale = None
        self.encoder = model.visual
        embed = 640 if args.model_name =='RN50x4' else 512
        self.fc_head = nn.Linear(embed, num_classes, bias=False)
    def forward(self, x):
        
        image_features = self.encoder(x)
        #print(image_features.shape)
        logits = self.fc_head(image_features)
        return logits
        
class ViT_VPT_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(ViT_VPT_Model, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'RN50x4' if args.model_name =='RN50x4' else  'ViT-B/16'
        model = ViT_VPT(args, model_name, device)
        model_dict = model.state_dict()
        if args.model_name !='RN50x4':
            checkpoint = torch.load('./ViT-B_16.pt', map_location="cpu")
            new_checkpoint = {}
            for k in list(checkpoint.keys()):
                if k.startswith('visual.'):
                    new_checkpoint[k] = checkpoint[k]
            del checkpoint
            pos_embed_w = resize_pos_embed(new_checkpoint['visual.positional_embedding'],model.state_dict()['visual.positional_embedding'], num_tokens=1, gs_new=(18,18))
            new_checkpoint['visual.positional_embedding'] = pos_embed_w
            matched_dict = {k: v for k, v in new_checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
            print('matched keys:', len(matched_dict))
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            
        model.transformer = None
        model.token_embedding = None
        model.ln_final = None
        model.text_projection = None
        model.logit_scale = None
        self.encoder = model.visual
        embed = 640 if args.model_name =='RN50x4' else 512
        self.fc_head = nn.Linear(embed, num_classes, bias=False)
    def forward(self, x):
        
        image_features = self.encoder(x)
        #print(image_features.shape)
        logits = self.fc_head(image_features)
        return logits
        
        

class Swin_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(Swin_Model, self).__init__()    
        self.encoder = swin_base_patch4_window7_224_in22k()
        model_dict = self.encoder.state_dict()
        checkpoint = torch.load('./swin_base_patch4_window7_224_22k.pth',map_location=torch.device('cpu'))
        checkpoint = checkpoint['model']
        print('original keys:',len(checkpoint))
        matched_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict and v.shape==model_dict[k].shape:
                matched_dict[k] = v
            elif k in model_dict and v.shape!=model_dict[k].shape:
                #pass
                
                new_v_shape = model_dict[k].shape
                old_v = v
                if len(new_v_shape)==2:
                    oldv = old_v.reshape(1,1,old_v.shape[0],old_v.shape[1]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[0],new_v_shape[1]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                else:
                    oldv = old_v.reshape(1,old_v.shape[0],old_v.shape[1],old_v.shape[2]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[1],new_v_shape[2]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                matched_dict[k] = newv
                
        print('matched keys:',len(matched_dict))
        model_dict.update(matched_dict)
        self.encoder.load_state_dict(model_dict)
        self.fc_head = nn.Linear(1024, num_classes, bias=False)
        
    def forward(self, x):
        image_features = self.encoder(x)
        logits = self.fc_head(image_features)
        return logits


class VPT_swin_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(VPT_swin_Model, self).__init__()    
        self.encoder = VPT_swin_base_patch4_window7_224_in22k()
        model_dict = self.encoder.state_dict()
        checkpoint = torch.load('./swin_base_patch4_window7_224_22k.pth',map_location=torch.device('cpu'))
        checkpoint = checkpoint['model']
        print('original keys:',len(checkpoint))
        matched_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict and v.shape==model_dict[k].shape:
                matched_dict[k] = v
            elif k in model_dict and v.shape!=model_dict[k].shape:
                #pass
                
                new_v_shape = model_dict[k].shape
                old_v = v
                if len(new_v_shape)==2:
                    oldv = old_v.reshape(1,1,old_v.shape[0],old_v.shape[1]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[0],new_v_shape[1]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                else:
                    oldv = old_v.reshape(1,old_v.shape[0],old_v.shape[1],old_v.shape[2]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[1],new_v_shape[2]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                matched_dict[k] = newv
                
        print('matched keys:',len(matched_dict))
        model_dict.update(matched_dict)
        self.encoder.load_state_dict(model_dict)
        self.fc_head = nn.Linear(1024, num_classes, bias=False)
        
    def forward(self, x):
        image_features = self.encoder(x)
        logits = self.fc_head(image_features)
        return logits

class SSF_swin_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(SSF_swin_Model, self).__init__()    
        self.encoder = SSF_swin_base_patch4_window7_224_in22k()
        model_dict = self.encoder.state_dict()
        checkpoint = torch.load('./swin_base_patch4_window7_224_22k.pth',map_location=torch.device('cpu'))
        checkpoint = checkpoint['model']
        print('original keys:',len(checkpoint))
        matched_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict and v.shape==model_dict[k].shape:
                matched_dict[k] = v
            elif k in model_dict and v.shape!=model_dict[k].shape:
                #pass
                
                new_v_shape = model_dict[k].shape
                old_v = v
                if len(new_v_shape)==2:
                    oldv = old_v.reshape(1,1,old_v.shape[0],old_v.shape[1]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[0],new_v_shape[1]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                else:
                    oldv = old_v.reshape(1,old_v.shape[0],old_v.shape[1],old_v.shape[2]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[1],new_v_shape[2]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                matched_dict[k] = newv
                
        print('matched keys:',len(matched_dict))
        model_dict.update(matched_dict)
        self.encoder.load_state_dict(model_dict)
        self.fc_head = nn.Linear(1024, num_classes, bias=False)
        
    def forward(self, x):
        image_features = self.encoder(x)
        logits = self.fc_head(image_features)
        return logits

class EVP_swin_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(EVP_swin_Model, self).__init__()    
        self.encoder = EVP_swin_base_patch4_window7_224_in22k()
        model_dict = self.encoder.state_dict()
        checkpoint = torch.load('./swin_base_patch4_window7_224_22k.pth',map_location=torch.device('cpu'))
        checkpoint = checkpoint['model']
        print('original keys:',len(checkpoint))
        matched_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict and v.shape==model_dict[k].shape:
                matched_dict[k] = v
            elif k in model_dict and v.shape!=model_dict[k].shape:
                #pass
                
                new_v_shape = model_dict[k].shape
                old_v = v
                if len(new_v_shape)==2:
                    oldv = old_v.reshape(1,1,old_v.shape[0],old_v.shape[1]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[0],new_v_shape[1]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                else:
                    oldv = old_v.reshape(1,old_v.shape[0],old_v.shape[1],old_v.shape[2]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[1],new_v_shape[2]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                matched_dict[k] = newv
                
        print('matched keys:',len(matched_dict))
        model_dict.update(matched_dict)
        self.encoder.load_state_dict(model_dict)
        self.fc_head = nn.Linear(1024, num_classes, bias=False)
        
    def forward(self, x):
        image_features = self.encoder(x)
        logits = self.fc_head(image_features)
        return logits



class Adapter_Swin_Model(nn.Module):
    def __init__(self, args, num_classes=5):
      
        super(Adapter_Swin_Model, self).__init__()    
        self.encoder = Adapter_swin_base_patch4_window7_224_in22k()
        model_dict = self.encoder.state_dict()
        checkpoint = torch.load('./swin_base_patch4_window7_224_22k.pth',map_location=torch.device('cpu'))
        checkpoint = checkpoint['model']
        print('original keys:',len(checkpoint))
        matched_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict and v.shape==model_dict[k].shape:
                matched_dict[k] = v
            elif k in model_dict and v.shape!=model_dict[k].shape:
                #pass
                
                new_v_shape = model_dict[k].shape
                old_v = v
                if len(new_v_shape)==2:
                    oldv = old_v.reshape(1,1,old_v.shape[0],old_v.shape[1]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[0],new_v_shape[1]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                else:
                    oldv = old_v.reshape(1,old_v.shape[0],old_v.shape[1],old_v.shape[2]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[1],new_v_shape[2]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                matched_dict[k] = newv
                
        print('matched keys:',len(matched_dict))
        model_dict.update(matched_dict)
        self.encoder.load_state_dict(model_dict)
        self.fc_head = nn.Linear(1024, num_classes, bias=False)
        
    def forward(self, x):
        image_features = self.encoder(x)
        logits = self.fc_head(image_features)
        return logits

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Swin_Seg_Model(nn.Module):
    def __init__(self, args, num_classes=5):
        super(Swin_Seg_Model, self).__init__()    
        #self.encoder = swin_base_patch4_window7_224_in22k(seg=True)
        #self.encoder = VPT_swin_base_patch4_window7_224_in22k(seg=True)
        #self.encoder = SSF_swin_base_patch4_window7_224_in22k(seg=True)
        #self.encoder = EVP_swin_base_patch4_window7_224_in22k(seg=True)
        self.encoder = Adapter_swin_base_patch4_window7_224_in22k(seg=True)
        model_dict = self.encoder.state_dict()
        checkpoint = torch.load('./swin_base_patch4_window7_224_22k.pth',map_location=torch.device('cpu'))
        checkpoint = checkpoint['model']
        print('original keys:',len(checkpoint))
        matched_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict and v.shape==model_dict[k].shape:
                matched_dict[k] = v
            elif k in model_dict and v.shape!=model_dict[k].shape:
                new_v_shape = model_dict[k].shape
                old_v = v
                if len(new_v_shape)==2:
                    oldv = old_v.reshape(1,1,old_v.shape[0],old_v.shape[1]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[0],new_v_shape[1]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                else:
                    oldv = old_v.reshape(1,old_v.shape[0],old_v.shape[1],old_v.shape[2]).float()
                    newv = F.interpolate(oldv, size=(new_v_shape[1],new_v_shape[2]), mode='bicubic', align_corners=False).long().squeeze().squeeze()
                matched_dict[k] = newv
                
        print('matched keys:',len(matched_dict))
        model_dict.update(matched_dict)
        self.encoder.load_state_dict(model_dict)
        
        # transformer decoder
        self.in_channels = [128,256,512,1024]
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.decoder_embedding_dim = 256
        self.decoder_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.decoder_embedding_dim)
        self.decoder_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.decoder_embedding_dim)
        self.decoder_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.decoder_embedding_dim)
        self.decoder_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.decoder_embedding_dim)
        self.decoder_pred = nn.Conv2d(self.decoder_embedding_dim*4, num_classes, kernel_size=1) 
        self.up = nn.Upsample(size=None, scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        c1, c2, c3, c5, c4 = self.encoder(x)
        n, _, h, w = c4.shape
        _c4 = self.decoder_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = nn.Upsample(size=c1.size()[2:], scale_factor=None, mode='bilinear', align_corners=False) (_c4)
        
        _c3 = self.decoder_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.Upsample(size=c1.size()[2:], scale_factor=None, mode='bilinear', align_corners=False) (_c3)
        
        _c2 = self.decoder_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = nn.Upsample(size=c1.size()[2:], scale_factor=None, mode='bilinear', align_corners=False) (_c2)
        
        _c1 = self.decoder_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        x = self.decoder_pred(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.up (x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
           
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = x1 + x2
        x = self.conv_bn_relu(x)
        return x  
    
class My_Decoder(nn.Module):
    def __init__(self,channels, num_classes): 
        super(My_Decoder, self).__init__()
        channels = channels
        self.decode4 = Decoder(channels[3],channels[2])
        self.decode3 = Decoder(channels[2],channels[1])
        self.decode2 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential( nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                      nn.Conv2d(channels[0], num_classes, kernel_size=1,bias=False))

    def forward(self, x):
        encoder = x
        d4 = self.decode4(encoder[3], encoder[2]) 
        d3 = self.decode3(d4, encoder[1]) 
        d2 = self.decode2(d3, encoder[0])
        out = self.decode0(d2)    
        return out
    
class RN50x4_UNet(nn.Module):
    def __init__(self,args,num_classes):
        super(RN50x4_UNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'RN50x4' if args.model_name =='RN50x4' else  'ViT-B/16'
        model, preprocess = clip.load(args, model_name,  device)
        model_dict = model.state_dict()
        checkpoint = torch.load('./RN50x4.pt', map_location="cpu")
        new_checkpoint = {}
        for k in list(checkpoint.keys()):
            if k.startswith('visual.'):
                new_checkpoint[k] = checkpoint[k]
        del checkpoint
        matched_dict = {k: v for k, v in new_checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
        print('matched keys:', len(matched_dict))
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)
        
        
        model.transformer = None
        model.token_embedding = None
        model.ln_final = None
        model.text_projection = None
        model.logit_scale = None
        model.visual.attnpool = None
        self.encoder = model.visual
        channels = [80,640,1280,2560]
        self.decoder = My_Decoder(channels, num_classes)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output   
    

    
