import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import glob
from tqdm import tqdm

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name,inx,batch_size=32):
        self.batch_size = batch_size
        # self.path = './dataset/'
        self.path = './dataset_train/'
        self.path2='./dataset_val/'
        self.num =len(glob.glob(self.path+'/*'))
        self.num2=len(glob.glob(self.path2+'/*'))
        self.dataset_name = dataset_name
        self.train_fn =glob.glob(self.path+'/*')
        self.test_fn=glob.glob(self.path2+'/*')
       


        self.load_data(inx)
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()
        
        

    def __len__(self):
        return len(self.meta_data)

    def load_data(self,i):
        self.train_data = []
        self.flow_data = []
        self.val_data = []

        
        # f = np.load(self.train_fn[i])
        im=np.load(self.train_fn[i])['i0i1gt']
        f=np.load(self.train_fn[i])['ft0ft1']
        self.train_data=im
        self.flow_data=f
        
        # # for i in range(100):
        # for i in tqdm(range(self.num)):
        #     f = np.load('dataset_train/{}.npz'.format(i))
        #     # if i < 80:
        #     self.train_data.append(f['i0i1gt'])
        #     self.flow_data.append(f['ft0ft1'])
        #     # else:
        #     #     self.val_data.append(f['i0i1gt'])
        
        # for i in tqdm(range(self.num2)):
        #     f = np.load('dataset_val/{}.npz'.format(i))
        #     self.val_data.append(f['i0i1gt'])
    
        if self.dataset_name == 'train':
            self.meta_data= self.train_data
            self.nr_sample = self.num
        else:
            self.meta_data = self.val_data
            self.nr_sample = self.num2
                

    

    def aug(self, img0, gt, img1, flow_gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow_gt = flow_gt[x:x+h, y:y+w, :]
        return img0, gt, img1, flow_gt

    def getimg(self, index):
        data = self.meta_data[index]
        img0 = data[0:3].transpose(1, 2, 0)
        img1 = data[3:6].transpose(1, 2, 0)
        gt = data[6:9].transpose(1, 2, 0)
        flow_gt = (self.flow_data[index]).transpose(1, 2, 0)
        return img0, gt, img1, flow_gt
            
    def __getitem__(self, index):        
        img0, gt, img1, flow_gt = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1, flow_gt = self.aug(img0, gt, img1, flow_gt, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                flow_gt = flow_gt[::-1]
                flow_gt = np.concatenate((flow_gt[:, :, 0:1], -flow_gt[:, :, 1:2], flow_gt[:, :, 2:3], -flow_gt[:, :, 3:4]), 2)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                flow_gt = flow_gt[:, ::-1]
                flow_gt = np.concatenate((-flow_gt[:, :, 0:1], flow_gt[:, :, 1:2], -flow_gt[:, :, 2:3], flow_gt[:, :, 3:4]), 2)
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                flow_gt = np.concatenate((flow_gt[:, :, 2:4], flow_gt[:, :, 0:2]), 2)
        else:
            flow_gt = np.zeros((256, 448, 4))
        flow_gt = torch.from_numpy(flow_gt.copy()).permute(2, 0, 1)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0), flow_gt

if __name__ == '__main__':
    VimeoDataset('train')