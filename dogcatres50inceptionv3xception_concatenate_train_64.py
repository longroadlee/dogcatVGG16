import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation,Dropout,Flatten,Dense,concatenate
from keras import optimizers
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image
from keras.callbacks import EarlyStopping

#res50 inputshape
height=224
width=224
#inceptionv3 and xception inputshape
height_inception=299
width_inception=299

num_train_dog=8000
num_train_cat=8000
num_val_dog=2483
num_val_cat=2482
batch_size=32
img_path_dog_train='/home/iacp/dogcatdata/train/dog'
img_path_cat_train='/home/iacp/dogcatdata/train/cat'
img_path_dog_val='/home/iacp/dogcatdata/validation/dog'
img_path_cat_val='/home/iacp/dogcatdata/validation/cat'

count=0
res_train_data=np.ndarray(shape=(num_train_dog+num_train_cat+1,height,width,3),dtype=np.uint8)
res_val_data=np.ndarray(shape=(num_val_dog+num_val_cat+1,height,width,3),dtype=np.uint8)
inception_train_data=np.ndarray(shape=(num_train_dog+num_train_cat+1,height_inception,width_inception,3),dtype=np.uint8)
inception_val_data=np.ndarray(shape=(num_val_dog+num_val_cat+1,height_inception,width_inception,3),dtype=np.uint8)
train_labels=np.zeros(shape=(num_train_dog+num_train_cat+1),dtype=np.uint8)
val_labels=np.zeros(shape=(num_val_dog+num_val_cat+1),dtype=np.uint8)

for i in os.listdir(img_path_dog_train):
    load_path=os.path.join(img_path_dog_train,i)
    img=image.load_img(load_path,target_size=(height,width))
    img=image.img_to_array(img)
    img_dog_res=np.expand_dims(img,axis=0)
    
    img=image.load_img(load_path,target_size=(height_inception,width_inception))
    img=image.img_to_array(img)
    img_dog_inception=np.expand_dims(img,axis=0)
    
    res_train_data[count,:]=preprocess_input(img_dog_res)
    inception_train_data[count,:]=preprocess_input(img_dog_inception)
    count=count+1
    if count>=num_train_dog:
        break
    
for i in os.listdir(img_path_cat_train):
    load_path=os.path.join(img_path_cat_train,i)
    img=image.load_img(load_path,target_size=(height,width))
    img=image.img_to_array(img)
    img_cat_res=np.expand_dims(img,axis=0)
    
    img=image.load_img(load_path,target_size=(height_inception,width_inception))
    img=image.img_to_array(img)
    img_cat_inception=np.expand_dims(img,axis=0)
    
    count=count+1
    res_train_data[count,:]=preprocess_input(img_cat_res)
    inception_train_data[count,:]=preprocess_input(img_cat_inception)
    train_labels[count]=1 #cat label:1
    if count>=num_train_dog+num_train_cat:
        break
    

count=0
for i in os.listdir(img_path_dog_val):
    load_path=os.path.join(img_path_dog_val,i)
    img=image.load_img(load_path,target_size=(height,width))
    img=image.img_to_array(img)
    img_dog_res=np.expand_dims(img,axis=0)
    
    img=image.load_img(load_path,target_size=(height_inception,width_inception))
    img=image.img_to_array(img)
    img_dog_inception=np.expand_dims(img,axis=0)
    
    res_val_data[count,:]=preprocess_input(img_dog_res)
    inception_val_data[count,:]=preprocess_input(img_dog_inception)
    count=count+1
    if count>=num_val_dog:
        break
    
for i in os.listdir(img_path_cat_val):
    load_path=os.path.join(img_path_cat_val,i)
    img=image.load_img(load_path,target_size=(height,width))
    img=image.img_to_array(img)
    img_cat_res=np.expand_dims(img,axis=0)
    
    img=image.load_img(load_path,target_size=(height_inception,width_inception))
    img=image.img_to_array(img)
    img_cat_inception=np.expand_dims(img,axis=0)
    
    count=count+1
    res_val_data[count,:]=preprocess_input(img_cat_res)
    inception_val_data[count,:]=preprocess_input(img_cat_inception)
    val_labels[count]=1 #cat label:1
    if count>=num_val_dog+num_val_cat:
        break
    
base_model=ResNet50(include_top=False,weights=None,input_shape=(height,width,3),pooling='avg')
base_model2=Xception(include_top=False,weights=None,input_shape=(height_inception,width_inception,3),pooling='avg')
base_model3=InceptionV3(include_top=False,weights=None,input_shape=(height_inception,width_inception,3),pooling='avg')

base_model.load_weights('./keras_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')#pls attention the location you run the script!!!!!!
base_model2.load_weights('./keras_weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')#pls attention the location you run the script!!!!!!
base_model3.load_weights('./keras_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')#pls attention the location you run the script!!!!!!
x=base_model.output
x2=base_model2.output
x3=base_model3.output

concat_x=concatenate([x,x2,x3])
x=Dropout(0.5)(concat_x)
x=Dense(4096,activation='relu')(x)
x=Dropout(0.5)(concat_x)
x=Dense(4096,activation='relu')(x)
pred=Dense(1,activation='sigmoid')(x)
model=Model(inputs=[base_model.input,base_model2.input,base_model3.input],outputs=pred)

for layer in base_model.layers:
    layer.trainable=False
for layer in base_model2.layers:
    layer.trainable=False
for layer in base_model3.layers: 
    layer.trainable=False

opt=optimizers.SGD(lr=0.001,momentum=0.9,nesterov=True)
model.compile(optimizer=opt,metrics=['binary_accuracy'],loss='binary_crossentropy')
model.fit([res_train_data,inception_train_data,inception_train_data],train_labels,validation_data=([res_val_data,inception_val_data,inception_val_data],val_labels),callbacks=[EarlyStopping(monitor='val_loss',patience=2,mode='min')],shuffle=True,batch_size=batch_size,epochs=20,verbose=1)
model.save_weights('dogcatres50inceptionv3xception_concatenate_train_32.h5')

