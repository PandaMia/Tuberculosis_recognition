import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self.create_conv_block(1, 32)
        self.block2 = self.create_conv_block(32, 64)
        self.block3 = self.create_conv_block(64, 128)
        self.block4 = self.create_conv_block(128, 256)

        self.classifier = nn.Sequential(nn.Linear(16384, 1024),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(1024),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(1024),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(1024, 256),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 1))

    def create_conv_block(self, in_channels, out_channels):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(1, 1), padding_mode='replicate'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(1, 1), padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels))
        return conv_block

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
