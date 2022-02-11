from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
import keras.backend as K
# from keras_lr_multiplier import LRMultiplier

'''
code source:
https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras

'''
import os
# os.system('export CUDA_VISIBLE_DEVICES=1')
os.environ['CUDA_VISIBLE_DEVICES']='1'
# inres=(256,256)
# num_channels=3
# bottleneck=bottleneck_mobile

def create_hourglass_network(num_classes, num_stacks, num_channels, inres, outres, bottleneck,inchannel):

    num_channels = int(num_channels)
    inchannel = int(inchannel)
    num_classes = int(num_classes)


    input = Input(shape=(int(inres[0]), int(inres[1]), inchannel))

    front_features = create_front_module(input, num_channels, bottleneck) #contain 3 res blocks

    head_next_stage = front_features
    
    # masked_person = Lambda(lambda a: a[:,:,:,:3])(input)
    # cloth = Lambda(lambda a: a[:,:,:,3:6])(input)

    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i,[input])
        # outputs.append(head_to_loss)
        outputs.append(head_to_loss)

    model = Model(inputs=input, outputs=outputs)
    rms = RMSprop(lr=5e-4)
    model.compile(optimizer=rms, loss=mean_squared_error, metrics=["accuracy"])

    #new code part
    #extra -- upto3rd layer lr different
    # model_temp = Model(inputs=input, outputs=outputs[:3])
    # trained_layers = model_temp.layers
    # dict1={}
    # for i in model.layers:
    #     if(i in trained_layers):
    #         dict1[i.name]=5e-4/2.0
    #     else:
    #         dict1[i.name]=5e-4


    # optimizer=LRMultiplier('RMSprop', dict1)
    # model.compile(optimizer=optimizer, loss=mean_squared_error, metrics=["accuracy"])


    return model#,model_temp


def hourglass_module(bottom, num_classes, num_channels, bottleneck, hgid,list_inputs):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(bottom, bottleneck, hgid, num_channels)

    # create right features, connect with left features
    rf1 = create_right_half_blocks(left_features, bottleneck, hgid, num_channels)

    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hgid, num_channels,list_inputs)

    return head_next_stage, head_parts


def bottleneck_block(bottom, num_out_channels, block_name):
    # skip layer
    if K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = Conv2D(num_out_channels, kernel_size=(1, 1), activation='tanh', padding='same',
                       name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = Conv2D(num_out_channels // 2, kernel_size=(1, 1), activation='tanh', padding='same',
                name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    

    _x = Conv2D(num_out_channels // 2, kernel_size=(3, 3), activation='tanh', padding='same',
                name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    

    _x = Conv2D(num_out_channels, kernel_size=(1, 1), activation='tanh', padding='same',
                name=block_name + '_conv_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    
    _x = Add(name=block_name + '_residual')([_skip, _x])

    return _x


def bottleneck_mobile(bottom, num_out_channels, block_name):
    # skip layer
    if K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='tanh', padding='same',
                                name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = SeparableConv2D(num_out_channels // 2, kernel_size=(1, 1), activation='tanh', padding='same',
                         name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    
    
    _x = SeparableConv2D(num_out_channels // 2, kernel_size=(3, 3), activation='tanh', padding='same',
                         name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    

    _x = SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='tanh', padding='same',
                         name=block_name + '_conv_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    
    _x = Add(name=block_name + '_residual')([_skip, _x])

    return _x


def create_front_module(input, num_channels, bottleneck,name_code='default'):
    # front module, input to 1/4 resolution
    # 1 7x7 conv + maxpooling
    # 3 residual block

    _x = Conv2D(64, kernel_size=(7, 7), strides=(1,1), padding='same', activation='tanh', name='front_conv_1x1_x1_%s'%(name_code))(
        input)
    _x = BatchNormalization()(_x)

    _x = bottleneck(_x, num_channels // 2, 'front_residual_x1_%s'%(name_code))
    # _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

    _x = bottleneck(_x, num_channels // 2, 'front_residual_x2_%s'%(name_code))
    _x = bottleneck(_x, num_channels, 'front_residual_x3_%s'%(name_code))

    # model1 = Model(inputs=[input],outputs=[_x])

    return _x


def create_left_half_blocks(bottom, bottleneck, hglayer, num_channels):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hgname = 'hg' + str(hglayer)

    f1 = bottleneck(bottom, num_channels, hgname + '_l1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

    f2 = bottleneck(_x, num_channels, hgname + '_l2')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

    f4 = bottleneck(_x, num_channels, hgname + '_l4')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

    f8 = bottleneck(_x, num_channels, hgname + '_l8')

    return (f1, f2, f4, f8)


def connect_left_to_right(left, right, bottleneck, name, num_channels):
    '''
    :param left: connect left feature to right feature
    :param name: layer name
    :return:
    '''
    # left -> 1 bottlenect
    # right -> upsampling
    # Add   -> left + right

    _xleft = bottleneck(left, num_channels, name + '_connect')
    _xright = UpSampling2D()(right)
    add = Add()([_xleft, _xright])
    out = bottleneck(add, num_channels, name + '_connect_conv')
    return out


def bottom_layer(lf8, bottleneck, hgid, num_channels):
    # blocks in lowest resolution
    # 3 bottlenect blocks + Add

    lf8_connect = bottleneck(lf8, num_channels, str(hgid) + "_lf8")

    _x = bottleneck(lf8, num_channels, str(hgid) + "_lf8_x1")
    _x = bottleneck(_x, num_channels, str(hgid) + "_lf8_x2")
    _x = bottleneck(_x, num_channels, str(hgid) + "_lf8_x3")

    rf8 = Add()([_x, lf8_connect])

    return rf8


def create_right_half_blocks(leftfeatures, bottleneck, hglayer, num_channels):
    lf1, lf2, lf4, lf8 = leftfeatures

    rf8 = bottom_layer(lf8, bottleneck, hglayer, num_channels)

    rf4 = connect_left_to_right(lf4, rf8, bottleneck, 'hg' + str(hglayer) + '_rf4', num_channels)

    rf2 = connect_left_to_right(lf2, rf4, bottleneck, 'hg' + str(hglayer) + '_rf2', num_channels)

    rf1 = connect_left_to_right(lf1, rf2, bottleneck, 'hg' + str(hglayer) + '_rf1', num_channels)

    return rf1


# def create_heads(prelayerfeatures, rf1, num_classes, hgid, num_channels):
#     # two head, one head to next stage, one head to intermediate features
#     head = Conv2D(num_channels, kernel_size=(1, 1), activation='tanh', padding='same', name=str(hgid) + '_conv_1x1_x1')(
#         rf1)
#     head = BatchNormalization()(head)

#     # for head as intermediate supervision, use 'linear' as activation.
#     head_parts = Conv2D(num_classes, kernel_size=(1, 1), activation='sigmoid', padding='same',
#                         name=str(hgid) + '_conv_1x1_parts')(head)
#     head_parts_debapriya = head_parts
#     # #moule to convert to coordinate from image 
#     # head_parts_debapriya = Flatten()(head_parts)
#     # # head_parts_debapriya = Dense(2048,activation = 'sigmoid')(head_parts_debapriya)
#     # # head_parts_debapriya = Dense(1024,activation = 'sigmoid')(head_parts_debapriya)
#     # head_parts_debapriya = Dense(1024,activation = 'sigmoid')(head_parts_debapriya)
#     # head_parts_debapriya = Dense(num_classes*2,activation = 'sigmoid')(head_parts_debapriya)

#     # use linear activation
#     head = Conv2D(num_channels, kernel_size=(1, 1), activation='sigmoid', padding='same',
#                   name=str(hgid) + '_conv_1x1_x2')(head)
#     head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='sigmoid', padding='same',
#                     name=str(hgid) + '_conv_1x1_x3')(head_parts)

#     head_next_stage = Add()([head, head_m, prelayerfeatures])
#     return head_next_stage, head_parts,head_parts_debapriya
def create_heads(prelayerfeatures, rf1, num_classes, hgid, num_channels,list_inputs):
    # two head, one head to next stage, one head to intermediate features
    # [masked_person] = list_inputs
    head = Conv2D(num_channels, kernel_size=(3, 3), activation='tanh', padding='same', name=str(hgid) + '_conv_1x1_x1')(
        rf1)
    head = BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    output_img = Conv2D(3, kernel_size=(1, 1), activation='sigmoid', padding='same',
                        name=str(hgid)+'out_im' + '_conv_3x3_parts')(head)
    # output_mask = Conv2D(1, kernel_size=(1, 1), padding='same',
    #                     name=str(hgid)+'out_mask' + '_conv_1x1_parts', activation='sigmoid',kernel_regularizer=regularizers.l1(0.01))(head)
    # output_mask1 = Lambda(lambda a: K.repeat_elements(a,3,axis=-1))(output_mask)
    # # #mask from image
    # output_cl_mask = Conv2D(1, kernel_size=(1, 1), padding='same',
    #                     name=str(hgid)+'out_mask_cl' + '_conv_1x1_parts', activation='sigmoid',kernel_regularizer=regularizers.l1(0.01))(head)
    # output_cl_mask1 = Lambda(lambda a: K.repeat_elements(a,3,axis=-1))(output_cl_mask)

    # #mask from cloth
    # output_img_mask = Conv2D(1, kernel_size=(1, 1), padding='same',
    #                     name=str(hgid)+'out_mask_img' + '_conv_1x1_parts', activation='sigmoid',kernel_regularizer=regularizers.l1(0.01))(head)
    # output_img_mask1 = Lambda(lambda a: K.repeat_elements(a,3,axis=-1))(output_img_mask)
    

    # output_img_combi = Lambda( lambda d:  d[-1]*d[1] +  (1-d[1])*d[0]  )([masked_person,output_mask1,output_img]) 

    # output_img_combi = Lambda( lambda d:  d[-1]*d[1] +  (1-d[1])*d[0]  )([masked_person,output_img_mask1,output_img]) 
    # output_img_combi = Lambda( lambda d:  d[-1]*d[1] +  (1-d[1])*d[0]  )([cloth,output_cl_mask1,output_img_combi]) 

    # output_img_combi = Lambda( lambda d:  d[0]*d[-1] + d[]  )([masked_person,output_img,cloth,output_cl_mask1,output_img_mask1]) 

    # head_parts = Concatenate(axis=-1)([output_img,output_mask,output_img_combi])

    # head_parts = Concatenate(axis=-1)([output_img,output_cl_mask,output_img_mask,output_img_combi])
    head_parts = output_img
    # head_parts = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
    #                     name=str(hgid) + '_conv_1x1_parts')(head)

    # use linear activation
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                  name=str(hgid) + '_conv_1x1_x2')(head)
    head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                    name=str(hgid) + '_conv_1x1_x3')(head_parts)

    head_next_stage = Add()([head, head_m, prelayerfeatures])
    return head_next_stage, head_parts

def euclidean_loss(x, y):
    return K.sqrt(K.sum(K.square(x - y)))
