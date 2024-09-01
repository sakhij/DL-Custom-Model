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
    # Load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # One hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# Scale pixels
def prep_pixels(train, test):
    # Convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # Normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # Return normalized images
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
    # Plot loss
    pyplot.figure(figsize=(12, 6))
    
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()

    # Plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()

    # Save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.show()
    pyplot.close()

# Run the test harness for evaluating a model
def run_test_harness():
    # Load dataset
    trainX, trainY, testX, testY = load_dataset()
    # Prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # Define model
    model = define_model()
    
    # Initialize lists to keep track of metrics
    epoch_metrics = []

    # Define a custom callback to log details for each epoch
    class EpochLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # Store logs for current epoch
            epoch_metrics.append({
                'epoch': epoch + 1,
                'loss': logs.get('loss'),
                'accuracy': logs.get('accuracy'),
                'val_loss': logs.get('val_loss'),
                'val_accuracy': logs.get('val_accuracy')
            })

    # Fit model with the custom callback
    history = model.fit(trainX, trainY, epochs=20, batch_size=64, validation_data=(testX, testY), verbose=0, callbacks=[EpochLogger()])

    # Print detailed metrics for each epoch
    for metrics in epoch_metrics:
        epoch = metrics['epoch']
        print(f"Epoch {epoch} - loss: {metrics['loss']:.4f}, accuracy: {metrics['accuracy']:.4f}, val_loss: {metrics['val_loss']:.4f}, val_accuracy: {metrics['val_accuracy']:.4f}")
        
        # Predict on the test set
        y_pred_prob = model.predict(testX, batch_size=64, verbose=0)
        
        # Convert probabilities to class labels
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Convert true labels to class labels
        y_true = np.argmax(testY, axis=1)
        
        # Calculate precision and recall for each class
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        
        print(f" \t- precision: {precision:.4f}, recall: {recall:.4f}")
        print()
    
    # Evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    
    print(f'Test Accuracy: {acc:.4f}')
    
    # Learning curves
    summarize_diagnostics(history)
    
    # Save model
    model_path = 'cifar10_model.h5'
    model.save(model_path)
    print(f'Model saved to {model_path}')

# Entry point, run the test harness
run_test_harness()
