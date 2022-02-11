import sys
sys.path.append('./')
import params
dataset_path = params.dataset_path
import os 
from PIL import Image
# import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy.misc import imresize,imread
# ----------------------------------------

def get_reshaped(im):
	if(len(im.shape)==3):
		im_new = np.ones((256,256,3),dtype='uint8')
	else:
		im_new = np.ones((256,256),dtype='uint8')
	im_new= im_new*im[0][0]
	im_new[:,32:-32] = im
	return im_new

from models.skin_detection import model_skin_detector

skin_model, skin_model_color,_ = model_skin_detector.build_network()
skin_model.load_weights('./checkpoints/skin_detection/%d_wt_1.h5'%(49999))

def generate(seg_choice=False,filename='%sdeepfashion_names_upper.txt'%('train_test_files/'), shuffle=True,batch_size=5,type='train'):

	f = open(filename,'r')
	lines = f.readlines()

	
	i=-1;j=0
	tot_lines = len(lines)
	if(shuffle==True):
		np.random.shuffle(lines)

	l_zzero = np.zeros(shape=(batch_size,256,256,1),dtype='float32')
	
	while True:
			
		l_seg = [];l_name=[]
		l_im = []		
		# l_z = []
		batch_count = 0
		
		while(1):

			i = (i+1)%(tot_lines)
			
			line = lines[i][:-1]
			sname = line.split(' ')[0]

			l_name.append(sname.split('/')[-1])
			sname = dataset_path+sname
			
			im = np.array(Image.open(sname))
			im = imresize(im,(256,256))

			if(seg_choice==False):
				seg  = skin_model.predict([im[np.newaxis,:,:,:]/255.0])[0][0]
				ret,thresh1 = cv2.threshold((seg*255).astype('uint8'),127,255,cv2.THRESH_BINARY)

				seg = thresh1[:,:,:1]/255.0
			else:				
				mname = sname.replace('.png','_mask.jpg')
				seg = np.array(Image.open(mname))
				seg = (imresize(seg,(256,256))[:,:,np.newaxis])/255.0

			l_im.append(im/255.0)
			l_seg.append(seg)
			
			batch_count+=1		   
			if(batch_count==batch_size):
				break		

		yield ([l_im,l_seg,l_zzero,l_name])