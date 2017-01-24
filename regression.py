import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential  # 按顺序
from keras.layers import Dense   #全连接层
import matplotlib as mil
mil.use('TkAgg')

import matplotlib.pyplot as plt


# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()


X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points

#build a neural network
model = Sequential()
model.add(Dense(output_dim= 1, input_dim= 1 ))

# loss function and optimizing methond
model.compile(loss='mse',optimizer='sgd')


# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train) # 默认返回值 is cost
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()