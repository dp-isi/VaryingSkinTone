import sys
sys.path.append('./')
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from models import stage1_model

gan, discriminator,generator = stage1_model.build_network()

from data import data_loader_test as data_loader

import argparse
def str2bool(value):
    return value.lower() == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int,default=1,help='your batch size')
parser.add_argument('-range_count', type=int,default=1,help='how many skin tones for a single image')
parser.add_argument('-seg_choice', type=str2bool,default=False,help='if segmentation is provided externally provide input should be True else False')
parser.add_argument('-test_filename', type=str,default='test.txt',help='test file name')

args = parser.parse_args()



test_filename = args.test_filename
bs=args.batch_size
range_count=args.range_count
seg_choice=args.seg_choice

f = open('./train_test_files/%s'%(test_filename))
ds = len(f.readlines())
f.close()
# ------------------------------------------------------


obj = data_loader.generate(seg_choice=seg_choice, filename = './train_test_files/%s'%(test_filename),batch_size=bs,shuffle=False)

# -------------------------------------------------------
gan.load_weights('./checkpoints/skin_model/final_%d_wt.h5'%(9000))

# -------------------------------------------------------
paths='./results/'
path_im=paths
if(not os.path.exists(path_im)):
	os.mkdir(path_im)


tones = np.linspace(-0.15,0.25,range_count)

def f(im):
	return (im*255).astype('uint8')

for ii in range(ds):

	im_b,seg_b,z0_b,name_b = next(obj)

	for i in range(bs):

		im,seg,z0,name = im_b[i:i+1],seg_b[i:i+1],z0_b[i:i+1],name_b[i:i+1]
		seg = np.array(seg)

		Image.fromarray(f(im[0])).save(path_im+'%s_%d_gt.jpg'%(name[0],0))
		Image.fromarray(f(seg[0][:,:,0])).save(path_im+'%s_%d_seg.png'%(name[0],0))

		for j in tones:
			print(j)
			
			var = j*seg
	
			res = gan.predict([im,seg,z0 + var,z0])
		
			imn0 = res[0][0][:,:,:3]

			Image.fromarray(f(imn0)).save(path_im+'%s_%f.jpg'%(name[0],j))
