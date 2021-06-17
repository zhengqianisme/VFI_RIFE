#!/usr/bin/env python

import torch

import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy
import PIL.Image
import glob
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append('/media/gky-u/DATA/z/arXiv-2020-RIFE')
# from model.liteflownet.liteflownetv1 import liteflownet
# from model.RAFT.raftv2 import raftmodel


def flow_model(flow_name,img0,img1):
	if flow_name == "liteflownet":
		return liteflownet(img0, img1)
	if flow_name == "raft":
		return raftmodel(img0, img1)

def create_list(path,file_name_new,num,data,imgflo='/i0i1gt.npy'):
	if os.path.exists(file_name_new):
		os.remove(file_name_new)
	
	with open(file_name_new,'a') as f:
		
		# for file_name in tqdm(glob.glob(path+'/*')):
		for i in range(0,num):
			file_name=str(data)+str(i)+str(imgflo)
			f.write(file_name+'\n')
		


if __name__ == '__main__':
	
	path='/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/zq/rife_t/dataset_train'
	file_name_new='trainlist.txt'
	data='./dataset_train/'
	num=len(glob.glob(path+'/*'))
	create_list(path,file_name_new,num,data,imgflo='/i0i1gt.npy')

	path='/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/zq/rife_t/dataset_train'
	file_name_new='flowlist.txt'
	data='./dataset_train/'
	num=len(glob.glob(path+'/*'))
	imgflo = '/ft0ft1.npy'
	create_list(path,file_name_new,num,data,imgflo)
	
	# path='/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/zq/rife_t/dataset_val'
	# file_name_new='vallist.txt'
	# data='./dataset_val/'
	# num2=len(glob.glob(path+'/*'))
	# create_list(path,file_name_new,num2,data,imgflo='/i0i1gt.npy')

	################################################################################################################
	# origin dont change
	# dirtop="dataset_train"
	# dirtop_val="dataset_val"
	# path = 'vimeo_interp_test/'
	# f = open(path + 'tri_trainlist.txt', 'r')
	# f_val = open(path + 'tri_testlist.txt', 'r')
	# # f=['00001/0001','00001/0002']
	# # f_val=f
	# numimg=11389
	# for i in f:
	# 	name = str(i).strip()
	# 	if (len(name) <= 1):
	# 		continue

	# 	print(path + 'target/' + name )

	# 	I0 = path + 'target/' + name + '/im1.png'
	# 	I1 = path + 'target/' + name + '/im2.png'
	# 	I2 = path + 'target/' + name + '/im3.png'

	# 	# flo0 = './model/liteflownet/images/first.png'
	# 	# # flo1 = './model/liteflownet/images/second.png'
	# 	# f0 = liteflownet(I0, I2)
	# 	# f1 = liteflownet(I2, I0)

	# 	f0=flow_model("raft",I0, I2)
	# 	f1=flow_model("raft",I2, I0)
	# 	# f0=flow_model("liteflownet",I0, I2)
	# 	# f1=flow_model("liteflownet",I2, I0)
	# 	f = numpy.concatenate((f0, f1), 0)

	# 	I0 = torch.FloatTensor(numpy.ascontiguousarray(
	# 		numpy.array(PIL.Image.open(I0))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)))
	# 	I1 = torch.FloatTensor(numpy.ascontiguousarray(
	# 		numpy.array(PIL.Image.open(I1))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)))
	# 	I2 = torch.FloatTensor(numpy.ascontiguousarray(
	# 		numpy.array(PIL.Image.open(I2))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)))
	# 	i = numpy.concatenate((I0, I1, I2), 0)

		
	# 	if not os.path.exists(dirtop):
	# 		os.mkdir(dirtop)

	# 	dir=str(dirtop)+"/"+str(numimg)+".npz"

	# 	# numpy.save(str(numimg)+"/i0i1gt.npy", i)
	# 	numpy.savez(dir,i0i1gt=i,ft0ft1=f)
	# 	numimg+=1


	# numimg=0
	# for i in f_val:
	# 	name = str(i).strip()
	# 	if (len(name) <= 1):
	# 		continue

	# 	print(path + 'target/' + name )

	# 	I0 = path + 'target/' + name + '/im1.png'
	# 	I1 = path + 'target/' + name + '/im2.png'
	# 	I2 = path + 'target/' + name + '/im3.png'

	# 	f0 = liteflownet(I0, I2)
	# 	f1 = liteflownet(I2, I0)
	# 	f = numpy.concatenate((f0, f1), 0)

	# 	I0 = torch.FloatTensor(numpy.ascontiguousarray(
	# 		numpy.array(PIL.Image.open(I0))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)))
	# 	I1 = torch.FloatTensor(numpy.ascontiguousarray(
	# 		numpy.array(PIL.Image.open(I1))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)))
	# 	I2 = torch.FloatTensor(numpy.ascontiguousarray(
	# 		numpy.array(PIL.Image.open(I2))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)))
	# 	i = numpy.concatenate((I0, I1, I2), 0)

		
	# 	if not os.path.exists(dirtop_val):
	# 		os.mkdir(dirtop_val)

	# 	dir=str(dirtop_val)+"/"+str(numimg)+".npz"

	# 	# numpy.save(str(numimg)+"/i0i1gt.npy", i)
	# 	numpy.savez(dir,i0i1gt=i,ft0ft1=f)
	# 	numimg+=1

	# print("ok")