import json
import argparse
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from cnn_model import CNN


class Training(object):

    def __init__(self, device, epochs, lr, batch_size):
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.params_file = './source/data/data_params.json'
        self.save_path = './source/model/model_weights.pth'
        self.train_dir, self.val_dir = None, None
        self.img_shape = None
        self.mean = None
        self.std = None
        self.train_loader, self.val_loader = None, None
        self.model, self.criterion, self.optimizer, self.scheduler = None, None, None, None
        self.best_state = None
        self.history = {'train_loss': [], 'train_accuracy': [], 'train_f1': [],
                        'val_loss': [], 'val_accuracy': [], 'val_f1': []}
        self.load_params()
        self.create_model()
        self.create_loaders()

    def load_params(self):
        with open(self.params_file, 'r') as f:
            params = json.loads(f.read())
        self.train_dir = params['train']
        self.val_dir = params['val']
        self.img_shape = params['img_shape']
        self.mean = params['mean']
        self.std = params['std']

    def create_model(self):
        self.model = CNN().to(self.device)
        pos_weight = torch.tensor([5]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                    mode='min',
                                                                    patience=2,
                                                                    factor=0.4,
                                                                    verbose=True)

    def create_loaders(self):
        train_transforms = transforms.Compose([transforms.Resize(self.img_shape),
                                               transforms.Grayscale(num_output_channels=1),
                                               transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                               transforms.RandomAffine(degrees=0,
                                                                       translate=(0.1, 0.1),
                                                                       scale=(0.9, 1.1)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=self.mean, std=self.std)])
        val_transforms = transforms.Compose([transforms.Resize(self.img_shape),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=self.mean, std=self.std)])

        train_dataset = datasets.ImageFolder(self.train_dir, train_transforms)
        val_dataset = datasets.ImageFolder(self.val_dir, val_transforms)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        min_loss = float('inf')

        for epoch in range(self.epochs):
            print('Epoch %02d' % (epoch+1))
            train_loss, train_accuracy, train_f1 = self.fit_epoch()
            val_loss, val_accuracy, val_f1 = self.evaluate()
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_f1'].append(val_f1)

            print('Train_loss: %.4f, Train_acc: %.4f, Train_f1: %.4f | Val_loss: %.4f, Val_acc: %.4f, Val_f1: %.4f' \
                  % (train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1))

            if val_loss < min_loss:
                self.best_state = deepcopy(self.model.state_dict())
                min_loss = val_loss

    def fit_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_f1 = 0.0
        processed_size = 0

        for inputs, labels in tqdm(self.train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.float32)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            preds = torch.round(torch.sigmoid(outputs.detach()).squeeze(1)).cpu()
            labels = labels.cpu()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += accuracy_score(labels, preds) * inputs.size(0)
            running_f1 += f1_score(labels, preds, zero_division=1) * inputs.size(0)
            processed_size += inputs.size(0)

        train_loss = running_loss / processed_size
        train_accuracy = running_accuracy / processed_size
        train_f1 = running_f1 / processed_size
        return train_loss, train_accuracy, train_f1

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        processed_size = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.float32)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                preds = torch.round(torch.sigmoid(outputs.detach()).squeeze(1)).cpu()
                labels = labels.cpu()

                running_loss += loss.item() * inputs.size(0)
                processed_size += inputs.size(0)

                all_labels.append(labels)
                all_preds.append(preds)

        val_loss = running_loss / processed_size

        all_labels = torch.hstack(all_labels)
        all_preds = torch.hstack(all_preds)

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, zero_division=1)
        return val_loss, val_accuracy, val_f1

    def save_model(self):
        self.model.load_state_dict(self.best_state)
        torch.save(self.model.state_dict(), self.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs of training (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate (default: 0.0003)')
    parser.add_argument('--bs', type=int, default=32, help='Batch size (default: 32)')

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.lr
    batch_size = args.bs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t = Training(device, epochs, lr, batch_size)
    t.train()
    t.save_model()
