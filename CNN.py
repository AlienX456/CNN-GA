from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras

#Model imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization

from sklearn.metrics import classification_report
#get_ipython().run_line_magic('matplotlib', 'inline')


#   self.data_train = pd.read_csv('data/fashionmnist/fashion-mnist_train.csv')
#   self.data_test = pd.read_csv('data/fashionmnist/fashion-mnist_test.csv')
#   img_rows, img_cols = 28, 28
#   batch 256
#   classes 10

class CNN:
    def __init__(self, num_conv_layers, data_train_path, data_test_path, img_rows, img_cols, epoch_number, batch_size, num_classes):
        self.num_conv_layers = num_conv_layers
        self.data_train = pd.read_csv(data_train_path)
        self.data_test = pd.read_csv(data_test_path)
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.input_shape = (self.img_rows, self.img_cols, 1)
        self.epoch_number = epoch_number
        self.num_classes = num_classes
        self.batch_size = batch_size

        #data variables
        self.X, self.y, self.X_train, self.X_val, self.y_train, self.y_val = None,None,None,None,None,None

    def configure_data(self):

        self.X = np.array(self.data_train.iloc[:, 1:])
        self.y = to_categorical(np.array(self.data_train.iloc[:, 0]))

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=13)

        #Test data
        self.X_test = np.array(self.data_test.iloc[:, 1:])
        self.y_test = to_categorical(np.array(self.data_test.iloc[:, 0]))

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.img_rows, self.img_cols, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.img_rows, self.img_cols, 1)
        self.X_val = self.X_val.reshape(self.X_val.shape[0], self.img_rows, self.img_cols, 1)

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_val = self.X_val.astype('float32')
        self.X_train /= 255
        self.X_test /= 255
        self.X_val /= 255

    #propose a solution to the input size problem on each layer
    def training_cnn(self, num_filters, size_filters):
        model = None

        history = None

        model = Sequential()

        model.add(Conv2D(num_filters[0], kernel_size=(size_filters[0], size_filters[0]),
                         activation='relu',
                         kernel_initializer='he_normal',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(num_filters[1], (size_filters[1], size_filters[1]), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(num_filters[2], (size_filters[2], size_filters[2]), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(self.X_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epoch_number,
                  verbose=1,
                  validation_data=(self.X_val, self.y_val))
        score = model.evaluate(self.X_test, self.y_test, verbose=0)

        print('Train accuracy', history.history['acc'])
        print('Train loss', history.history['loss'])
        print('Val accuracy : ',history.history['val_acc'])
        print('Val loss : ',history.history['val_loss'])
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return [model,history,score]

    def generate_precision_graph(self, history):
        accuracy = history.history['acc']
        val_accuracy = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(accuracy))
        plt.figure()
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    def generate_classification_report(self, model):
        predicted_classes = model.predict_classes(self.X_test)

        #get the indices to be plotted
        y_true = self.data_test.iloc[:, 0]
        correct = np.nonzero(predicted_classes==y_true)[0]
        incorrect = np.nonzero(predicted_classes!=y_true)[0]

        target_names = ["Class {}".format(i) for i in range(self.num_classes)]
        print(classification_report(y_true, predicted_classes, target_names=target_names))

        plt.figure()
        for i, correct in enumerate(correct[:9]):
            plt.subplot(3, 3, i + 1)
            plt.imshow(self.X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))
            plt.tight_layout()

        plt.figure()
        for i, incorrect in enumerate(incorrect[0:9]):
            plt.subplot(3,3,i+1)
            plt.imshow(self.X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))
            plt.tight_layout()



#########################################################################################
#############JUST FOR TEXT PURPOSES######################################################

def test():
    #def __init__(self, num_conv_layers, data_train_path, data_test_path, img_rows, img_cols, epoch_number, batch_size, num_classes):
    cnn = CNN(3, 'data/fashionmnist/fashion-mnist_train.csv', 'data/fashionmnist/fashion-mnist_test.csv', 28, 28, 3, 256, 10)
    cnn.configure_data()
    cnn.training_cnn([32, 64, 128], [15, 5, 1])
    cnn.generate_classification_report()

if __name__ == "__main__":
	test()
