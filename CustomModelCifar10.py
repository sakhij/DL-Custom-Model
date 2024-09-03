import sys
import keras
from matplotlib import pyplot
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.metrics import precision_score, recall_score
from keras.datasets import cifar10

# Load train and test dataset
def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# Scale pixels
def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

# Define CNN model
def define_model():
    model = Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Plot diagnostic learning curves
def summarize_diagnostics(history):
    pyplot.figure(figsize=(12, 6))
    
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()

    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + 'plots.png')
    pyplot.show()
    pyplot.close()

# Run the test harness for evaluating a model
def run_test_harness():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    model = define_model()
    history = model.fit(trainX, trainY, epochs=20, batch_size=64, validation_data=(testX, testY), verbose=0)
    _, acc = model.evaluate(testX, testY, verbose=0)

    y_pred_prob = model.predict(testX, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(testY, axis=1)
    
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    print(f'Precision Metric: {precision:.4f}')
    print(f'Recall Metric: {recall:.4f}')
    summarize_diagnostics(history)
    
    model_path = 'cifar10_model.h5'
    model.save(model_path)
    print(f'Model saved to {model_path}')

run_test_harness()
