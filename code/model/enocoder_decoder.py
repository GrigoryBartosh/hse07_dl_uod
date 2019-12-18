import torch
from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # N x M x params
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # for im in encoded:
        #     decoded.append(self.decoder(im)[None, :, :, :])
        return decoded
