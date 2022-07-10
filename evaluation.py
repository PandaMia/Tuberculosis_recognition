import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from cnn_model import CNN


class Evaluator(object):

    def __init__(self, device, batch_size):
        self.device = device
        self.batch_size = batch_size
        self.params_file = './source/data/data_params.json'
        self.weights_path = './source/model/model_weights.pth'
        self.test_dir = None
        self.img_shape = None
        self.mean = None
        self.std = None
        self.test_loader = None
        self.model, self.criterion = None, None
        self.load_params()
        self.create_model()
        self.create_loader()

    def load_params(self):
        with open(self.params_file, 'r') as f:
            params = json.loads(f.read())
        self.test_dir = params['test']
        self.img_shape = params['img_shape']
        self.mean = torch.tensor(params['mean'])
        self.std = torch.tensor(params['std'])

    def create_model(self):
        self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device(self.device)))
        pos_weight = torch.tensor([5]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def create_loader(self):
        test_transforms = transforms.Compose([transforms.Resize(self.img_shape),
                                              transforms.Grayscale(num_output_channels=1),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=self.mean, std=self.std)])

        test_dataset = datasets.ImageFolder(self.test_dir, test_transforms)

        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        processed_size = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.float32)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                preds = torch.round(torch.sigmoid(outputs.detach()).squeeze(1)).cpu()
                labels = labels.cpu()

                running_loss += loss.item() * inputs.size(0)
                processed_size += inputs.size(0)

                all_labels.append(labels)
                all_preds.append(preds)

        test_loss = running_loss / processed_size

        all_labels = torch.hstack(all_labels)
        all_preds = torch.hstack(all_preds)

        test_accuracy = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, zero_division=1)

        print(f'Loss: {test_loss:.4f}')
        print(f'Accuracy: {test_accuracy:.4f}'),
        print(f'F1-score: {test_f1:.4f}')

        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, figsize=(5, 5), colorbar=True, cmap='Spectral',
                              class_names=['Normal', 'Tuberculosis'])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--bs', type=int, default=32, help='Batch size (default: 32)')

    args = parser.parse_args()
    batch_size = args.bs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    e = Evaluator(device, batch_size)
    e.evaluate()
