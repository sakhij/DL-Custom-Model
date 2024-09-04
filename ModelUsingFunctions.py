import numpy as np
import requests
import tarfile
import os
import pickle
import matplotlib.pyplot as plt

# Function to download and extract the CIFAR-10 dataset
def download_and_extract_cifar10(url, extract_path):
    response = requests.get(url, stream=True)
    file_name = os.path.join(extract_path, 'cifar-10-python.tar.gz')
    with open(file_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=extract_path)

    print("Download and extraction complete.")

# Function to load CIFAR-10 dataset
def load_cifar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(os.path.join(cifar10_dataset_folder_path, 'data_batch_' + str(batch_id)), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    
    return features, np.array(labels)

def load_cifar10_data(cifar10_dataset_folder_path):
    X, y = [], []
    for i in range(1, 6):
        features, labels = load_cifar10_batch(cifar10_dataset_folder_path, i)
        X.append(features)
        y.append(labels)
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    with open(os.path.join(cifar10_dataset_folder_path, 'test_batch'), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    
    test_X = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_y = np.array(batch['labels'])
    
    return X, y, test_X, test_y

cifar10_dir = './cifar-10-batches-py'
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
download_and_extract_cifar10(url, './')

# Load data
train_X, train_y, test_X, test_y = load_cifar10_data(cifar10_dir)

# Normalize datatrain_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0

# One-hot encode labels
train_y = np.eye(10)[train_y]
test_y = np.eye(10)[test_y]

# Flatten the input data for the neural network
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

class my_NN(object):
    def __init__(self):
        self.input = 3072
        self.output = 10
        self.hidden_units = 512
        
        np.random.seed(1)
        self.w1 = np.random.randn(self.input, self.hidden_units) * 0.01
        self.w2 = np.random.randn(self.hidden_units, self.output) * 0.01

    def _forward_propagation(self, X):
        self.z2 = np.dot(X, self.w1)
        self.a2 = self._sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = self._softmax(self.z3)
        return self.a3

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _loss(self, predict, y):
        m = y.shape[0]
        logprobs = -np.log(predict[range(m), np.argmax(y, axis=1)])
        loss = np.sum(logprobs) / m
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        m = X.shape[0]
        delta3 = predict - y
        self.dw2 = np.dot(self.a2.T, delta3) / m
        delta2 = np.dot(delta3, self.w2.T) * self._sigmoid_prime(self.z2)
        self.dw1 = np.dot(X.T, delta2) / m

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _update(self, learning_rate=0.01):
        self.w1 -= learning_rate * self.dw1
        self.w2 -= learning_rate * self.dw2

    def train(self, X, y, iteration=100):
        losses = []
        accuracies = []
        for i in range(iteration):
            y_hat = self._forward_propagation(X)
            loss = self._loss(y_hat, y)
            self._backward_propagation(X, y)
            self._update()
            losses.append(loss)
            
            accuracy = self.score(np.argmax(y_hat, axis=1), y)
            accuracies.append(accuracy)
            
            if i % 10 == 0:
                print(f"Epoch {i} - Loss: {loss}, Accuracy: {accuracy}%")
        
        return losses, accuracies

    def predict(self, X):
        y_hat = self._forward_propagation(X)
        return np.argmax(y_hat, axis=1)

    def score(self, predict, y):
        accuracy = np.mean(predict == np.argmax(y, axis=1)) * 100
        return accuracy

    def save_model(self, file_path):
        model_data = {
            "w1": self.w1,
            "w2": self.w2
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            model_data = pickle.load(file)
        self.w1 = model_data["w1"]
        self.w2 = model_data["w2"]
        print(f"Model loaded from {file_path}")

if __name__ == '__main__':
    clf = my_NN()
    losses, accuracies = clf.train(train_X, train_y)
    
    clf.save_model('my_nn_model.pkl')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
  
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.show()

    # Evaluate on the test set
    pre_y = clf.predict(test_X)
    score = clf.score(pre_y, test_y)
    print('Predicted labels: ', pre_y[:10])
    print('True labels:', np.argmax(test_y, axis=1)[:10])
    print('Accuracy score: ', score)

    # Load the trained model and evaluate again
    clf.load_model('my_nn_model.pkl')
    pre_y = clf.predict(test_X)
    score = clf.score(pre_y, test_y)
    print('Accuracy score after loading the model: ', score)
