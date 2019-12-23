import os

import torch
import torchvision
from torch.utils.data import DataLoader

from code.dataset import EmojiDataset
from code.model.decoder import Decoder
from code.model.encoder import SSD300
from code.model.enocoder_decoder import EncoderDecoder


class Pipeline:
    def __init__(self, device):
        # GLOBAL SETTINGS
        self.data_path = '../datasets/mover'

        # Params
        self.device = device
        self.class_num = 4
        self.encoder_learning_rate = 0.0001
        self.batch_size = 2
        self.num_epochs = 50

        self.train_data = self.get_train_data()
        self.test_data = self.get_test_data()

        self.model = self.get_model()
        self.criterion = self.get_criterion()
        self.optimizer = self.get_encoder_optimizer()
        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()

    def get_model(self):
        encoder = SSD300(self.class_num + 1)
        decoder = Decoder(self.device, n_images=self.class_num)
        return EncoderDecoder(encoder, decoder)

    def get_criterion(self):
        return torch.nn.MSELoss()

    def get_encoder_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.encoder_learning_rate)

    def get_train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    @staticmethod
    def duplicate_data_loader(data_loader):
        for data in data_loader:
            yield data, data

    def get_train_data(self):
        return EmojiDataset(
            root=os.path.join(self.data_path, 'train/data'),
            transform=torchvision.transforms.ToTensor()
        )

    def get_test_data(self):
        return EmojiDataset(
            root=os.path.join(self.data_path, 'test/data'),
            transform=torchvision.transforms.ToTensor()
        )