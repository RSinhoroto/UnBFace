# import general and image processing libraries
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, show
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

# import keras modules
from keras.models import Model
from keras.optimizers import SGD
from keras.applications import InceptionV3
from keras.layers import Dense, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping

# import cyclic learning rate implementation
from clr_callback import CyclicLR

def plot_training(history, ymin=0, ymax=50):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].plot(history.history['dense_2_mae'], 
                 label='Yaw Train MAE')
    axes[0].plot(history.history['val_dense_2_mae'], 
                 label='Yaw Val MAE')
    axes[0].set_xlabel('Epochs')
    ymax = 2*np.min(history.history['val_dense_2_mae'])
    axes[0].set_ylim([ymin,ymax])
    axes[0].legend()
    
    axes[1].plot(history.history['dense_4_mae'], 
                 label='Pitch Train MAE')
    axes[1].plot(history.history['val_dense_4_mae'], 
                 label='Pitch Val MAE')
    axes[1].set_xlabel('Epochs')
    ymax = 2*np.min(history.history['val_dense_4_mae'])
    axes[1].set_ylim([ymin,ymax])
    axes[1].legend()

    axes[2].plot(history.history['dense_6_mae'], 
                 label='Roll Train MAE')
    axes[2].plot(history.history['val_dense_6_mae'], 
                 label='Roll Val MAE')
    axes[2].set_xlabel('Epochs')
    ymax = 2*np.min(history.history['val_dense_6_mae'])
    axes[2].set_ylim([ymin,ymax])
    axes[2].legend()
    
    return fig


timestamp = time.strftime('%Y_%m_%d_%Hh%Mm')

data_file = open('aflw_crops_poses_radians.txt','r')
data_file_content = data_file.readlines()

filenames = []
poses = []
for line in data_file_content:
  name, pose = line.split(':')
  filenames.append(name)
  poses.append(pose)

# load AFLW
face_data = []
eye_data = []
data_target = []

for i in range(len(filenames)):
# get pose
  current_pose = poses[i][1:-2]
  angles = current_pose.split(', ')
  data_target.append([float(angles[0]), 
                      float(angles[1]), 
                      float(angles[2])])

# load correspondent face crop
  im_path = "crops/" + filenames[i]
  img = imread(im_path, as_gray=False, plugin='matplotlib')
  face_data.append(img)

# load correspondent eye crop
  im_path = "eyes/" + filenames[i]
  img = imread(im_path, as_gray=False, plugin='matplotlib')
  eye_data.append(img)

  if i % 1000 == 0:
      print(str(i) + " images processed.")

# convert to np array
face_data = np.asarray(face_data)
eye_data = np.asarray(eye_data)

# load AFW

data_file = open('afw_crops_poses_radians.txt','r')
data_file_content = data_file.readlines()

filenames = []
poses = []
for line in data_file_content:
  name, pose = line.split(':')
  filenames.append(name)
  poses.append(pose)


afw_face_data = []
afw_eye_data = []
afw_data_target = []

for i in range(len(filenames)):
# get pose
  current_pose = poses[i][1:-2]
  angles = current_pose.split(', ')
  afw_data_target.append([float(angles[0]), 
                          float(angles[1]), 
                          float(angles[2])])

# load correspondent face crop
  im_path = "crops/" + filenames[i]
  img = imread(im_path, as_gray=False, plugin='matplotlib')
  afw_face_data.append(img)

# load correspondent eye crop
  im_path = "eyes/" + filenames[i]
  img = imread(im_path, as_gray=False, plugin='matplotlib')
  afw_eye_data.append(img)


face_data = np.append(face_data, np.asarray(afw_face_data), axis=0)
eye_data = np.append(eye_data, np.asarray(afw_eye_data),axis=0)
data_target = np.append(data_target, afw_data_target, axis=0)

face_train, face_test, eye_train, eye_test, train_target, test_target = train_test_split(
    face_data, eye_data, data_target, test_size=0.1, random_state=42 )

del face_data
del eye_data
del data_target

# declare instances of base InceptionV3(GoogLeNet) model
base_model = InceptionV3(include_top=False, 
                         pooling='avg', input_shape=(96,96,3))
base_model_eye = InceptionV3(include_top=False, 
                             pooling='avg', input_shape=(76,152,3))

# rename to use two instances of InceptionV3
for layer in base_model_eye.layers:
    layer.name = 'eye_' + layer.name

# add fully connect regression layers
main_out = base_model.output 
eye_out = base_model_eye.output
out = concatenate([main_out, eye_out]) 

#yaw
yaw = Dense(256, activation='relu')(out) 
yaw = Dense(1, activation='linear')(yaw)

#pitch
pitch = Dense(256, activation='relu')(out) 
pitch = Dense(1, activation='linear')(pitch)

#roll
roll = Dense(256, activation='relu')(out) 
roll = Dense(1, activation='linear')(roll)


model = Model(inputs=[base_model.input, 
                      base_model_eye.input], 
              outputs=[yaw, pitch, roll])

# configure some hyper parameters
INIT_LR = 5e-3
EPOCHS = 100
BATCH_SIZE = 96
#STEPS_PER_EPOCH = 320,
VALIDATION_STEPS = 64

# add cyclical learning rate callback
MIN_LR = 1e-7
MAX_LR = 1e-2
CLR_METHOD = "triangular"
STEP_SIZE = 4

clr = CyclicLR(mode=CLR_METHOD,
               base_lr=MIN_LR,
               max_lr=MAX_LR,
               step_size=(STEP_SIZE * (np.shape(face_train)[0] // BATCH_SIZE)))

# add checkpoint to save the network and stop if training doesn't improve
filepath = "../best_weights_" + timestamp + ".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='val_loss', patience=50)
callbacks_list = [checkpoint, earlystop, clr]

# compile complete model with optmizer and print summary on screen
optim = SGD(lr=INIT_LR, momentum=0.9)
model.compile(optimizer=optim,
              loss='mean_squared_error',
              metrics=['mae'])

with open("../model_summary_" + timestamp + ".txt", 'w') as f:
    with redirect_stdout(f):
        model.summary()

# train model and plot training history
history = model.fit(x=[face_train, eye_train],
                    y=[np.transpose(train_target)[0],
                       np.transpose(train_target)[1],
                       np.transpose(train_target)[2],], 
                    epochs=EPOCHS,
                    validation_data=([face_test, eye_test],
                                     [np.transpose(test_target)[0],
                                      np.transpose(test_target)[1],
                                      np.transpose(test_target)[2],]
                                     ),
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks_list)

history_plot = plot_training(history)
history_plot.savefig("../training_history_" + timestamp)