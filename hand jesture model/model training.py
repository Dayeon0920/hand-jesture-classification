import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
 
 
path_dir1 = './hand jesture model/dataset/one hand-one'
path_dir2 = './hand jesture model/dataset/one hand-two'
path_dir3 = './hand jesture model/dataset/one hand-palm'
path_dir4 = './hand jesture model/dataset/one hand-hand knife'
path_dir5 = './hand jesture model/dataset/one hand-fist'
path_dir6 = './hand jesture model/dataset/two hand-left fist'
path_dir7 = './hand jesture model/dataset/two hand-right fist'
path_dir8 = './hand jesture model/dataset/two hand-left up'
path_dir9 = './hand jesture model/dataset/two hand-right up'
path_dir10 = './hand jesture model/dataset/two hand-twist hand'
 
file_list1 = os.listdir(path_dir1) # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)
file_list3 = os.listdir(path_dir3)
file_list4 = os.listdir(path_dir4)
file_list5 = os.listdir(path_dir5)
file_list6 = os.listdir(path_dir6)
file_list7 = os.listdir(path_dir7)
file_list8 = os.listdir(path_dir8)
file_list9 = os.listdir(path_dir9)
file_list10 = os.listdir(path_dir10)
 
# train용 이미지 준비
num = 0;
train_img = np.float32(np.zeros((10000, 224, 224, 3))) # 394+413+461
train_label = np.float64(np.zeros((10000, 1)))
 
for img_name in file_list1:
    img_path = path_dir1+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1
 
for img_name in file_list2:
    img_path = path_dir2+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 1 # paper
    num = num + 1
 
for img_name in file_list3:
    img_path = path_dir3+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 2 # scissors
    num = num + 1

for img_name in file_list4:
    img_path = path_dir4+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1

for img_name in file_list5:
    img_path = path_dir5+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1

for img_name in file_list6:
    img_path = path_dir6+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1


for img_name in file_list7:
    img_path = path_dir7+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1

for img_name in file_list8:
    img_path = path_dir8+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1

for img_name in file_list9:
    img_path = path_dir9+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1

for img_name in file_list10:
    img_path = path_dir10+'/'+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1

# 이미지 섞기
     
n_elem = train_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
train_label = train_label[indices]
train_img = train_img[indices]
 
#%% 
# create the base pre-trained model
IMG_SHAPE = (224, 224, 3)
 
base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))
 
GAP_layer = GlobalAveragePooling2D()
dense_layer = Dense(3, activation=tf.nn.softmax)
 
model = Sequential([
        base_model,
        GAP_layer,
        dense_layer
        ])
 
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
 
model.fit(train_img, train_label, epochs=1)
 
# save model
model.save("./hand jesture model/model.h5")
 
print("Saved model to disk")  