from .hourglass_relu.hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile
from keras.layers import *
from keras.models import Model
from keras.initializers import RandomNormal
from keras import optimizers
from keras_contrib.losses import DSSIMObjective
from keras.activations import relu
import tensorflow as tf

from keras.applications.vgg19 import VGG19

def build_discriminator_patchgan(df=12):

    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_shape=(256,256)
    gen_out = Input(shape=img_shape+(3,))
    
    d1 = d_layer(gen_out, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(d4)
    
    model = Model([gen_out], [validity])
    model.compile(loss=[ 'mse'], optimizer=optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))#optimizer='sgd')

    return model

from .skin_detection import model_skin_detector
model_c = model_skin_detector.get_model_PP()
model_c.load_weights('checkpoints/skin_color_detection/%d_color_model_wt_1.h5'%(49999))

def loss(layer):

	def custom_loss(y_true,y_pred):

		m = 0.004
		eps=0.0002
		lambd = 0.003		

		gen_out = y_pred[:,:,:,:3]
		gen_out_z0 = y_pred[:,:,:,3:6]
		img_skinseg = y_pred[:,:,:,6:7]
		z = layer 

		img = y_true			
		
		model_c_func=  Model(inputs=model_c.inputs,outputs= [model_c.layers[-1].output,model_c.layers[-2].output,model_c.layers[-3].output,\
			model_c.layers[-4].output,model_c.layers[-5].output,model_c.layers[-6].output,\
			model_c.layers[-7].output])
		
		feat_generator = model_c_func([gen_out,img_skinseg])
		feat_generator0 = model_c_func([gen_out_z0,img_skinseg])
		feat_gt = model_c_func([img,img_skinseg])

		w = [2.0, 1.0, 1.0, 1.0, 1.0, 0.5,0.5]


		color_dist = w[0]*K.mean(K.square(feat_generator[0]-feat_gt[0]))

	 	#---------------------------------------------------------------------------------------------------------
		# '''
		vgg19=VGG19()
		outer_model = Model(inputs=[(vgg19).layers[0].input],outputs=[(vgg19).layers[2].output, (vgg19).layers[5].output,(vgg19).layers[10].output,
			(vgg19).layers[15].output,(vgg19).layers[13].output, (vgg19).layers[8].output])


		p_generator = outer_model([tf.image.resize_images(gen_out_z0, (224,224))])
		p_gt = outer_model([tf.image.resize_images(img, (224,224))])
		
		
		w = [1.0/K.prod(K.cast(vgg19.layers[2].output.get_shape().as_list()[1:],'float32')),\
		1.0/K.prod(K.cast(vgg19.layers[5].output.get_shape().as_list()[1:],'float32')),\
		1.0/K.prod(K.cast(vgg19.layers[10].output.get_shape().as_list()[1:],'float32')),\
		1.0/K.prod(K.cast(vgg19.layers[15].output.get_shape().as_list()[1:],'float32')),\
		1.0/K.prod(K.cast(vgg19.layers[13].output.get_shape().as_list()[1:],'float32')),\
		1.0/K.prod(K.cast(vgg19.layers[8].output.get_shape().as_list()[1:],'float32'))] 
		

		percep_dist = w[0]*K.sum(K.square(p_generator[0]-p_gt[0])) + w[1]*K.sum(K.square(p_generator[1]-p_gt[1])) + \
		w[2]*K.sum(K.square(p_generator[2]-p_gt[2])) + w[3]*K.sum(K.square(p_generator[3]-p_gt[3])) + w[4]*K.sum(K.square(p_generator[4]-p_gt[4]))+\
			w[5]*K.sum(K.square(p_generator[5]-p_gt[5]))
		
		p_generator1 = outer_model([tf.image.resize_images(gen_out_z0*(1-K.tile(img_skinseg,(1,1,1,3))), (224,224))])
		p_gt1 = outer_model([tf.image.resize_images(img*(1-K.tile(img_skinseg,(1,1,1,3))), (224,224))])

		percep_dist_non_skin = w[0]*K.sum(K.square(p_generator1[0]-p_gt1[0])) + w[1]*K.sum(K.square(p_generator1[1]-p_gt1[1])) + \
		w[2]*K.sum(K.square(p_generator1[2]-p_gt1[2])) + w[3]*K.sum(K.square(p_generator1[3]-p_gt1[3])) + w[4]*K.sum(K.square(p_generator1[4]-p_gt1[4]))+\
			w[5]*K.sum(K.square(p_generator1[5]-p_gt1[5]))
		loss_2 = percep_dist + percep_dist_non_skin + lambd * (  m * ( K.sum(z)/K.sum(img_skinseg) ) + K.log((0.5-color_dist)) - eps )
		lossval = K.mean(loss_2)		
		
		return lossval

	return custom_loss


import keras.backend as K
from keras.losses import mse

def build_upscale_z():

	latDim =1
	ip1=Input(shape=(latDim,))
	dcGenInput = ip1

	x=Dense(128*4*4, activation='relu')(dcGenInput)
	x=BatchNormalization(momentum=0.9)(x)
	x=LeakyReLU(0.1)(x)
	x=Reshape((4, 4, 128))(x)

	x=Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
	x=BatchNormalization(momentum=0.9)(x)
	x=LeakyReLU(0.1)(x)

	x=Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
	x=BatchNormalization(momentum=0.9)(x)
	x=LeakyReLU(0.1)(x)

	x=Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
	x=BatchNormalization(momentum=0.9)(x)
	x=LeakyReLU(0.1)(x)

	x=Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
	x=Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
	dcGenFinal=Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)
	dcGen=Model([ip1], dcGenFinal)
	# dcGen.summary()
	return dcGen

def build_network():

	hourglass = create_hourglass_network(num_classes=1,num_stacks = 1, num_channels = 128,\
	 									inres = (256,256), outres = (256,256), bottleneck = bottleneck_mobile,inchannel = 5)
	
	img_shape=(256,256)
	img = Input(shape=img_shape+(3,))
	img_skinseg = Input(shape=img_shape+(1,))
	zc=1
	z = Input(shape=img_shape+(zc,))
	
	z0 = Input(shape=img_shape+(zc,))
	model_upscale_z = build_upscale_z()
	discriminator = build_discriminator_patchgan()
	

	x = Concatenate()([img,z,img_skinseg])
	x0 = Concatenate()([img,z0,img_skinseg])

	gen_out = hourglass([x])
	gen_out0 = hourglass([x0])

	gen_out_merge = Concatenate()([gen_out,gen_out0,img_skinseg])

	
	gen_out_11 = Lambda(lambda a: a)(gen_out)	
	
	gen_out0_11 = Lambda(lambda a: a)(gen_out0)

	dis_out = discriminator([gen_out_11])
	dis_out0 = discriminator([gen_out0_11])

	discriminator.trainable=False
	
	gan = Model([img,img_skinseg,z,z0],[gen_out_merge,dis_out,dis_out0])


	gan.compile(loss = [loss(z),mse,mse] , optimizer=optimizers.Adam(lr=0.0006, beta_1=0.5, beta_2=0.999))
	generator = Model([img,img_skinseg,z,z0],[gen_out_merge])

	generator.compile(loss = [loss(z)] , optimizer=optimizers.Adam(lr=0.0006, beta_1=0.5, beta_2=0.999))
	
	return gan, discriminator,generator