import torch
import numpy as np
import torch.nn
import os

from PIL import Image


def load_sample(number):
    return Image.open(os.path.join('emoji', f'test_{number}.png'))


class Decoder(torch.nn.Module):
    def __init__(self, image_shape=(4, 300, 300), emoji_shape=(4, 64, 64), n_images=13, generate=True):
        super(Decoder, self).__init__()
        self.image_shape = image_shape
        self.emoji_shape = emoji_shape
        self.n_images = n_images
        self.generate = True

        self.images = torch.ones((n_images, *emoji_shape), requires_grad=True, dtype=torch.float32)

        if generate:
            for i in range(n_images):
                image = load_sample(i)
                image = torch.from_numpy(np.array(image)).float() / 255
                self.images[i] = image.permute(2, 0, 1)

    def get_image(self, ind):
        return self.images[ind]

    def forward(self, data):
        x = torch.round(data[:, 0] * self.image_shape[1]).int()
        y = torch.round(data[:, 1] * self.image_shape[2]).int()
        h = torch.round(data[:, 2] * self.image_shape[1]).int()
        w = torch.round(data[:, 3] * self.image_shape[2]).int()
        d = data[:, 4]
        c = data[:, 5:]

        image = torch.ones(*self.image_shape, dtype=torch.float32)

        n_samples = data.shape[0]
        order = np.arange(n_samples)[np.argsort(d)]
        for i in order:
            x1, y1, x2, y2 = x[i] - h[i] // 2, y[i] - w[i] // 2, x[i] + (h[i] + 1) // 2, y[i] + (w[i] + 1) // 2

            new_image = self.get_image(c[i].argmax())
            c_new = new_image[:3, :, :]
            alpha_new = new_image[3, :, :].repeat(3, 1, 1)
            c_old = image[:3, x1:x2, y1:y2]
            alpha_old = image[3, x1:x2, y1:y2].repeat(3, 1, 1)
            alpha_0 = alpha_new + alpha_old * (1 - alpha_new)

            image[:3, x1:x2, y1:y2] = (c_new * alpha_new + c_old * alpha_old * (1 - alpha_new)) / alpha_0
            image[3, x1:x2, y1:y2] = alpha_0[0, :, :]

        return image
