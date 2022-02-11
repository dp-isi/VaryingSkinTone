import sys
sys.path.append('./skin_detection')
from .hourglass_relu.hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile
from keras.layers import *
from keras.models import Model
from keras.initializers import RandomNormal
from keras import optimizers
from keras_contrib.losses import DSSIMObjective
from keras.activations import relu
import tensorflow as tf
from keras.losses import mse
from keras.applications.vgg19 import VGG19

def custom_loss(y_true,y_pred):

	loss_1 = DSSIMObjective(kernel_size = 5)(y_true,y_pred)
	vgg19=VGG19()
	# outer_model = Model(inputs=[(vgg19).layers[0].input],outputs=[(vgg19).layers[13].output, (vgg19).layers[8].output,(vgg19).layers[2].output])
	outer_model = Model(inputs=[(vgg19).layers[0].input],outputs=[(vgg19).layers[2].output, (vgg19).layers[5].output,(vgg19).layers[10].output,
		(vgg19).layers[15].output,(vgg19).layers[13].output, (vgg19).layers[8].output])

	y_true = tf.image.resize_images(y_true, (224,224))
	y_pred = tf.image.resize_images(y_pred, (224,224))

	feat_generator = outer_model([y_pred])
	feat_gt = outer_model([y_true])

	w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	style_loss = w[0]*K.mean(K.square(feat_generator[0]-feat_gt[0])) + w[1]*K.mean(K.square(feat_generator[1]-feat_gt[1])) + \
	w[2]*K.mean(K.square(feat_generator[2]-feat_gt[2])) + w[3]*K.mean(K.square(feat_generator[3]-feat_gt[3])) + w[4]*K.mean(K.square(feat_generator[4]-feat_gt[4]))+\
		w[5]*K.mean(K.square(feat_generator[5]-feat_gt[5]))



	count_loss = K.abs(K.sum(K.sum(K.sum(y_true,axis=-1),axis=-1),axis=-1) -  K.sum(K.sum(K.sum(y_pred,axis=-1),axis=-1),axis=-1))
	loss_value = tf.Variable(1.0)
	loss_2 = K.mean(style_loss)
	loss_1 = K.mean(loss_1)
	loss_3 = K.mean(count_loss)

	loss  = loss_1 + loss_2 + loss_3

	# loss  = 0.01*loss_1 + loss_2
	return loss


def custom_count_norm_mse_loss(y_true,y_pred):

	seg_gt, res_gt = y_true[:,:,:,3:],y_true[:,:,:,:3] 
	 
	seg, res = y_pred[:,:,:,3:],y_pred[:,:,:,:3] 
	
	count_gt = K.abs(K.sum(K.sum(K.sum(seg_gt,axis=-1),axis=-1),axis=-1))
	count = K.abs(K.sum(K.sum(K.sum(seg,axis=-1),axis=-1),axis=-1))

	loss_1 = K.sum(K.sum(K.square(res*seg - res_gt*seg_gt), axis=-1))/(count_gt+1)

	loss = K.mean(loss_1)
	return loss

def custom_msssim_loss(y_true,y_pred):
	
	loss = (1-(K.mean(tf.image.ssim_multiscale(y_true, y_pred, 1, filter_size=5,
    filter_sigma=1.5, k1=0.01, k2=0.03))))*100
	return loss


def get_model_PP():

	img = Input(shape=(256,256,3))
	mask = Input(shape=(256,256,1))
	x = Concatenate(axis=-1)([img, mask])

	_x = Conv2D(16,(3,3),strides=2,activation = 'relu')(x)
	_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
	_x = Conv2D(8,(3,3),strides=2,activation = 'relu')(x)
	_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
	_x = Conv2D(2,(3,3),strides=2,activation = 'relu')(x)
	_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
	_x = Conv2D(3,(3,3),strides=2,activation = 'relu')(_x)
	_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
	_x = Conv2D(3,(3,3),strides=2,activation = 'relu')(_x)
	_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
	_x = Conv2D(3,(3,3),strides=2,activation = 'sigmoid')(_x)
	model = Model([img,mask],[_x])
	# model.summary()
	return model

import keras.backend as K

def build_network(get_color_model=0):
	hourglass = create_hourglass_network(num_classes=1,num_stacks = 1, num_channels = 128,\
	 									inres = (256,256), outres = (256,256), bottleneck = bottleneck_mobile,inchannel = 3)
	img_shape=(256,256)
	img = Input(shape=img_shape+(3,))
	
	PPN = get_model_PP()

	# img_rgb = Lambda(lambda c: c/(K.sum(c,axis=-1,keepdims=True)==False))(img)
	# img_rgb = img
	img_rgb = img#Lambda(lambda c: c[:,:,:,3:])(img)
	img_RGB = img#Lambda(lambda c: c[:,:,:,:3])(img)
	x = hourglass([img_rgb]) #segmentation
	x3 = Lambda(lambda a: K.repeat_elements(a,3,axis=-1))(x)

	c_RGB = PPN([img_RGB,x]) #output color RGB
	c_mat = Lambda(lambda c: c*K.ones(img_shape+(3,)))(c_RGB)
	o = Lambda( lambda d:  d[-1]*d[1] +  (1-d[1])*d[0]  )([img_RGB,x,c_mat]) 

	# concat_x_o = Concatenate(axis=-1)([o,x3])

	# skin_model = Model(inputs=[ img ], outputs=[ x3, concat_x_o ] )	
	skin_model = Model(inputs=[ img ], outputs=[ x3, o ] )	

	# skin_model.compile(loss = [custom_loss,DSSIMObjective(kernel_size=5)] , loss_weights = [1,1], optimizer=optimizers.Adam(lr=0.0006, beta_1=0.5, beta_2=0.999))
	skin_model.compile(loss = [custom_loss,custom_msssim_loss] , loss_weights = [1,1], optimizer=optimizers.Adam(lr=0.0006, beta_1=0.5, beta_2=0.999))
	
	skin_model_color = Model(inputs=[ img ], outputs=[ c_mat,c_RGB ] )	
	# skin_model_c = Model(inputs=[ img ], outputs=[ c ] )	

	skin_model_full = Model(inputs=[ img ], outputs=[ x3, o, c_mat,c_RGB ] )


	return skin_model, skin_model_color,skin_model_full