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

        locs, confs = encoded[:, :, :4], encoded[:, :, 4:]
        locs = torch.sigmoid(locs)
        locs = (locs + 1) / 2
        confs = confs.softmax(dim=2)
        encoded = torch.cat([locs, confs], dim=2)

        decoded = self.decoder(encoded)
        # for im in encoded:
        #     decoded.append(self.decoder(im)[None, :, :, :])
        return decoded
