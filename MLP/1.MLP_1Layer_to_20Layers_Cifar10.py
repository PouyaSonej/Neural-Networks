from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

# load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# show dataset
plt.figure(figsize=(25,4))
for i in range(16):
    plt.subplot(1,16,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(y_train[i])

# Convert class vectors to binary class matrices.
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# define model
def define_model(shape, num_layers):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=shape))
    model.add(keras.layers.Flatten())
    for l in range(num_layers):
        model.add(keras.layers.Dense(units=512, activation='relu'))    
    model.add(keras.layers.Dense(units=num_classes, activation='softmax'))

    return model

# evaluate models
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
for num_layers in range(0, 21):
    model = define_model(x_train[0].shape, num_layers)

    # compile model
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # train model
    history = model.fit(x_train, y_train,
                        batch_size=200,
                        epochs=20,
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        verbose=0)
  
    #
    train_loss.append(history.history['loss'][-1])
    valid_loss.append(history.history['val_loss'][-1])
    train_acc.append(history.history['accuracy'][-1])
    valid_acc.append(history.history['val_accuracy'][-1])

    print('layers = {}, params = {}, train_acc = {:4f}, val_acc = {:4f}'.format(num_layers+1, model.count_params(), train_acc[-1], valid_acc[-1]))

plt.figure()
plt.title('loss')
plt.plot(train_loss, 'r-x')
plt.plot(valid_loss, 'b-o')

plt.figure()
plt.title('accuracy')
plt.plot(train_acc, 'r-x')
plt.plot(valid_acc, 'b-o')

# plot confusion matrix
import sklearn.metrics

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print('Accuracy = {} %'.format(100 * np.sum(y_pred == y_true) / len(y_true)))

conf = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize=None)

plt.imshow(conf)
print(conf)
conf_norm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true')
print(conf_norm)
