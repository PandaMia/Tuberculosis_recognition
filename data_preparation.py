import os
import shutil
import zipfile
import argparse
import json
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class DataPreprocessor(object):

    def __init__(self, archive_path):
        self.archive_path = archive_path
        self.data_path = os.path.split(archive_path)[0] + '/unzipped_data'
        self.params_file = './source/data/data_params.json'
        self.classes = ['Normal', 'Tuberculosis']
        self.train_dir = os.path.normpath(os.path.join(self.data_path, 'train'))
        self.val_dir = os.path.normpath(os.path.join(self.data_path, 'val'))
        self.test_dir = os.path.normpath(os.path.join(self.data_path, 'test'))
        self.img_height = 128
        self.img_width = 128
        self.img_shape = (self.img_height, self.img_width)
        self.mean = None
        self.std = None

    def prepare_data(self):
        self.unzip_data()
        self.create_directory(self.train_dir)
        self.create_directory(self.val_dir)
        self.create_directory(self.test_dir)
        self.split_data()
        shutil.rmtree(os.path.join(self.data_path, 'TB_Chest_Radiography_Database'))
        self.compute_mean_std()
        self.write_params()

    def unzip_data(self):
        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)
        os.mkdir(self.data_path)

        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            for file in tqdm(zip_ref.infolist(), desc='Extracting data'):
                zip_ref.extract(file, self.data_path)

    def create_directory(self, dir_name):
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        for cls in self.classes:
            os.makedirs(os.path.join(dir_name, cls))

    def copy_images(self, images, source_dir, dest_dir):
        for img_name in tqdm(images, desc=f"{dest_dir.split(os.sep)[-2]}"):
            shutil.copy2(os.path.join(source_dir, img_name), dest_dir)

    def split_data(self):
        for cls in self.classes:
            print(f'Splitting {cls} class data:')
            class_path = os.path.join(self.data_path, 'TB_Chest_Radiography_Database', cls)
            train, val = train_test_split(os.listdir(class_path), test_size=0.2, shuffle=True, random_state=42)
            val, test = train_test_split(val, test_size=0.5, shuffle=True, random_state=42)

            self.copy_images(train, class_path, os.path.join(self.train_dir, cls))
            self.copy_images(val, class_path, os.path.join(self.val_dir, cls))
            self.copy_images(test, class_path, os.path.join(self.test_dir, cls))
        print('Data splitted!', end='\n\n')

    def compute_mean_std(self):
        image_transforms = transforms.Compose([transforms.Resize(self.img_shape),
                                               transforms.Grayscale(num_output_channels=1),
                                               transforms.ToTensor()])

        dataset = datasets.ImageFolder(self.train_dir, transform=image_transforms)

        loader = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=False)

        print('Compute mean and std on train set for data normalization.')
        mean = 0.0
        for data, _ in tqdm(loader, desc='Compute mean:'):
            num_samples = data.size(0)
            data = data.view(num_samples, data.size(1), -1)  # [b, c, w, h] -> [b, c, w * h]
            mean += data.mean(2).sum(0)
        self.mean = mean / len(loader.dataset)

        var = 0.0
        for data, _ in tqdm(loader, desc='Compute std:'):
            num_samples = data.size(0)
            data = data.view(num_samples, data.size(1), -1)
            var += ((data - self.mean.unsqueeze(1)) ** 2).sum([0, 2])
        self.std = torch.sqrt(var / (len(loader.dataset) * self.img_height * self.img_width))

    def write_params(self):
        param_dict = {'train': self.train_dir,
                      'val': self.val_dir,
                      'test': self.test_dir,
                      'img_shape': self.img_shape,
                      'mean': self.mean.tolist(),
                      'std': self.std.tolist()}

        with open(self.params_file, 'w') as f:
            f.write(json.dumps(param_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('archive_path', type=str, help='Path to zipped data')

    args = parser.parse_args()
    archive_path = args.archive_path

    dp = DataPreprocessor(archive_path)
    dp.prepare_data()
