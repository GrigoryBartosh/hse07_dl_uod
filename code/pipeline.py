import os

import torch
import torchvision
from torch.utils.data import DataLoader

from code.dataset import EmojiDataset
from code.model.decoder import Decoder
from code.model.encoder import SSD300
from code.model.enocoder_decoder import EncoderDecoder


class Pipeline:
    def __init__(self):
        # GLOBAL SETTINGS
        self.data_path = './generated_dataset/data'

        # Params
        self.class_num = 13
        self.encoder_learning_rate = 0.0001
        self.batch_size = 32
        self.num_epochs = 50

        self.train_data = self.get_train_data()
        self.test_data = self.get_test_data()

        self.model = self.get_model()
        self.optimizer = self.get_encoder_optimizer()
        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()

    def get_model(self):
        encoder = SSD300(self.class_num)
        decoder = Decoder(n_images=self.class_num)
        return EncoderDecoder(encoder, decoder)

    def get_encoder_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.encoder_learning_rate)

    def get_train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def get_train_data(self):
        return EmojiDataset(
            root=os.path.join(self.data_path, 'train'),
            transform=torchvision.transforms.ToTensor()
        )

    def get_test_data(self):
        return EmojiDataset(
            root=os.path.join(self.data_path, 'test'),
            transform=torchvision.transforms.ToTensor()
        )