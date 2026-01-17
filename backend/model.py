import numpy as np
import struct
from array import array
from zipfile import ZipFile
from torch import nn, optim, no_grad, float, save, load
from torch.utils.data import DataLoader, Dataset

def unzip(src_path, dest_path):
    with ZipFile(src_path, "r") as zObject:
        zObject.extractall(dest_path)

# Loader class was provided by the kaggle dataset maker
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CustomImageDataset(Dataset):
    def __init__(self, images, labels):
        self.img_labels = labels
        self.imgs = images

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.img_labels[idx]

def train():
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    loader = MnistDataloader("./data/unzipped/train-images.idx3-ubyte", "./data/unzipped/train-labels.idx1-ubyte", "./data/unzipped/t10k-images.idx3-ubyte", "./data/unzipped/t10k-labels.idx1-ubyte")
    train, test = loader.load_data()

    train_images, train_labels = train
    test_images, test_labels = test

    train_dataset = CustomImageDataset(np.array(train_images, np.float32), train_labels)
    test_dataset = CustomImageDataset(np.array(test_images, np.float32), test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    def train_loop(model, loss_fn, optimizer, dataloader):
        model.train()
        for i, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                loss, current = loss.item(), i * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

    def test_loop(model, loss_fn, dataloader):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        
        with no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
    for i in range(epochs):
        print(f"Epoch: {i + 1}\n-------------------------------")
        train_loop(model, loss_fn, optimizer, train_dataloader)
        test_loop(model, loss_fn, test_dataloader)
    print("Done!")
    
    save(model.state_dict(), "./model-data/model_weights.pth")

def main():
    # unzip("./data/data.zip", "./data/unzipped")
    train()

if __name__ == "__main__":
    main()