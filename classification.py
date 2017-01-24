import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize X_train[0] is a tuple
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
print(len(X_train),X_train[1],len(X_train[1]))
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

#build nerual network
# model = Sequential([
#     Dense(output_dim= 32,input_dim= 784), #32是 output
#     Activation('relu'),
#
#     Dense(output_dim= 10),
#     Activation('softmax')
#     ])
model = Sequential()
model.add(Dense(output_dim= 64,input_dim= 784))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax')) #0.9721

# define optimizer
rmsprop = RMSprop(lr=0.001)
model.compile(optimizer= rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train,y_train,batch_size=32,nb_epoch=5)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)