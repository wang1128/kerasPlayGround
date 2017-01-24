import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28) # -1 is the number of the examples
X_test = X_test.reshape(-1, 1,28, 28)
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

model = Sequential()

#conv layer
model.add(Convolution2D(
    nb_filter= 32, # 滤波器 生成32个新图片？
    nb_row= 5,
    nb_col= 5,
    border_mode='same', #padding mothod
    input_shape=(1,28,28) #1 高度 28*28
))
model.add(Activation('relu'))
#pooling layer
model.add(MaxPooling2D(
    pool_size= (2,2),
    strides= (2,2), #跳两步
    border_mode= 'same',
))

#Conv layer2 output shape(64,14,14)

model.add(Convolution2D(64,5,5,border_mode='same'))
model.add(Activation('relu'))

#pooling layer 2
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#fully connect layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, nb_epoch=5, batch_size=32,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)