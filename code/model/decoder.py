import torch
import numpy as np
import torch.nn
import os

from PIL import Image

from code.model.stacker import Stacker


def load_sample(number):
    return Image.open(os.path.join('./../datasets/emoji', f'test_{number}.png'))


class Decoder(torch.nn.Module):
    def __init__(self, image_shape=(4, 300, 300), emoji_shape=(4, 64, 64), n_images=13, generate=True):
        super(Decoder, self).__init__()
        self.image_shape = image_shape
        self.emoji_shape = emoji_shape
        self.n_images = n_images
        self.generate = True

        self.images = torch.ones((n_images + 1, *emoji_shape), requires_grad=True, dtype=torch.float32)

        if generate:
            for i in range(n_images):
                image = load_sample(i)
                image = torch.from_numpy(np.array(image)).float() / 255
                self.images[i] = image.permute(2, 0, 1)
            self.images[n_images] = (torch.ones(emoji_shape).float())
            self.images[n_images].permute(2, 0, 1)

    def get_image(self, ind):
        return self.images[ind]

    def forward(self, X):
        n_pictures = X.shape[0]
        n_emoji = X.shape[1]
        img = torch.ones(n_pictures, *self.image_shape, dtype=torch.float32)

        for image, data in zip(img, X):  # TODO(anyone): rewrite in vector terms?
            x = torch.round(data[:, 0] * self.image_shape[1]).int()
            y = torch.round(data[:, 1] * self.image_shape[2]).int()
            h = torch.round(data[:, 2] * self.image_shape[1]).int()
            w = torch.round(data[:, 3] * self.image_shape[2]).int()
            # h, w = torch.full(h.shape, 64).int(), torch.full(w.shape, 64).int()
        d = data[:, 4]
        c = data[:, 5:]

        # order = np.arange(n_emoji)[torch.argsort(d, dim=1)].T
        for i in np.arange(n_emoji)[torch.argsort(d)].reshape(-1):
            x1, y1, x2, y2 = x[i] - h[i] // 2, y[i] - w[i] // 2, x[i] + (h[i] + 1) // 2, y[i] + (w[i] + 1) // 2
            if x1 < 0 or y1 < 0 or x2 > self.image_shape[1] or y2 > self.image_shape[2]:
                continue

            new_image = self.get_image(c[i].argmax())
            c_new = new_image[:3, :, :]
            alpha_new = new_image[3, :, :].repeat(3, 1, 1)
            c_old = image[:3, x1:x2, y1:y2]
            alpha_old = image[3, x1:x2, y1:y2].repeat(3, 1, 1)
            alpha_0 = alpha_new + alpha_old * (1 - alpha_new)

            image[:3, x1:x2, y1:y2] = (c_new * alpha_new + c_old * alpha_old * (1 - alpha_new)) / alpha_0
            image[3, x1:x2, y1:y2] = alpha_0[0, :, :]

        return img[:, :3]
