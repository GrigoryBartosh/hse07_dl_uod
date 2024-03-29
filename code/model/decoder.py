import torch
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from PIL import Image


def load_sample(number):
    return Image.open(os.path.join('./../datasets/big-emoji', f'test_{number}.png'))


class Decoder(torch.nn.Module):
    def __init__(self, device, image_shape=(4, 32, 32), emoji_shape=(4, 64, 64), n_images=4, generate=True):
        super(Decoder, self).__init__()
        self.device = device
        self.image_shape = image_shape
        self.emoji_shape = emoji_shape
        self.n_images = n_images
        self.generate = True

        images = torch.ones((n_images + 1, *image_shape), dtype=torch.float32)
        if generate:
            for i in range(n_images):
                image = load_sample(i)
                image = torch.from_numpy(np.array(image)).float() / 255
                images[i] = image.permute(2, 0, 1)
            images[n_images] = (torch.ones(image_shape).float())
            images[n_images].permute(2, 0, 1)

        self.images = nn.Parameter(images)

    def get_image(self, ind):
        return self.images[ind]

    def forward(self, X):
        batch_size = X.shape[0]
        X = self.filter_sugggest(X)
        suggest_size = X.shape[1]

        params = X[:, :, :4]
        conf = X[:, :, 4:]
        classes = conf.shape[2]

        c1, c2, w, h = torch.chunk(params, 4, dim=2)
        x1 = c1 - w / 2
        x2 = c1 + w / 2
        y1 = c2 - h / 2
        y2 = c2 + h / 2
        transforms = torch.cat([x1, x2, y1, y2], dim=2)
        transforms = transforms.reshape(-1, 4)
        transforms = transforms.unsqueeze(1).repeat(1, conf.shape[2], 1)
        transforms = transforms.reshape(-1, 4)
        images = self.images.repeat(batch_size * suggest_size, 1, 1, 1, 1)
        images = images.reshape(-1, *self.image_shape)

        processed = self.apply_transform(images, transforms)
        processed = processed.reshape(batch_size, suggest_size, classes, -1).permute(3, 0, 1, 2)
        processed = processed * conf
        processed = processed.permute(1, 2, 3, 0).sum(dim=2).mean(dim=1)
        processed = processed.reshape(batch_size, *self.image_shape)

        return processed[:, :3]

    def forward1(self, X):
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

            new_image = self.get_image(0)
            c_new = new_image[:3, :, :]
            alpha_new = new_image[3, :, :].repeat(3, 1, 1)
            c_old = image[:3, x1:x2, y1:y2]
            alpha_old = image[3, x1:x2, y1:y2].repeat(3, 1, 1)
            alpha_0 = alpha_new + alpha_old * (1 - alpha_new)

            image[:3, x1:x2, y1:y2] = (c_new * alpha_new + c_old * alpha_old * (1 - alpha_new)) / alpha_0
            image[3, x1:x2, y1:y2] = alpha_0[0, :, :]

        return img[:, :3]

    def filter_sugggest(self, X):
        # return X[:, -190:]
        return X[:, -50:]
        # confs = X[:, :, 4:]
        # confs = confs.max(dim=2)[0]
        # confs = confs.argmax(dim=1)
        # res = X[range(X.shape[0]), confs]
        # return res[:, None, :]

    def get_images(self, X):
        return X[:, :, :4]

    def get_images(self, X):
        X = X[:, :, 5:].argmax(dim=2)
        # TODO
        return X

    # transform : row1, row2, col1, col2 -- coordinates [0 - image_size)
    def apply_transform(self, images, transforms):
        N, CH, image_row, image_col = images.shape
        assert (N == transforms.shape[0])
        assert (image_col == image_row)
        image_size = image_row
        image_WH = image_col * image_row
        rows1, rows2, cols1, cols2 = (transforms * image_size).T
        box_rows = rows2 - rows1
        box_cols = cols2 - cols1
        box_WHs = (box_rows * box_cols).int()
        T = get_transform_matrix(image_size, transforms)
        C = get_coordinates_matrix(image_size)
        TC = T @ C
        TC = TC[:, :2]
        TC = TC.repeat(image_WH, 1, 1, 1).permute(1, 0, 2, 3)  # N x image_WH x 2 x image_WH
        B = C[:2].repeat(image_WH, N, 1, 1).permute(1, 3, 2, 0)
        EPS = 1e-8
        A = (TC - B).pow(2).sum(2)
        A = -A
        A = A.softmax(dim=2)
        A = A.to(self.device)

        P = images.reshape(N, CH, -1).permute(0, 2, 1)
        return (A @ P).reshape(N, image_size, image_size, CH).permute(0, 3, 1, 2)


def get_transform_matrix(image_size, transforms):
    T = torch.zeros((transforms.shape[0], 3, 3))
    T[:, 0, 0] = (transforms[:, 1] - transforms[:, 0])
    T[:, 1, 1] = (transforms[:, 3] - transforms[:, 2])
    T[:, 2, 2] = 1
    T[:, 0, 2] = transforms[:, 0] * image_size
    T[:, 1, 2] = transforms[:, 2] * image_size
    return T


def get_coordinates_matrix(size):
    return torch.tensor(np.vstack([
        np.indices((size, size)).reshape(2, -1),
        np.ones(size * size)
    ])).float()


def get_box_coordinates_matrix(image_size, transforms):
    N = transforms.shape[0]
    no_mask = get_coordinates_matrix(image_size)[:2]
    no_mask = no_mask.repeat(N, 1, 1)
    masks = torch.full((N, 2, image_size, image_size), 1e9)
    for i in range(N):
        masks[i, :, transforms[i, 0]:transforms[i, 1], transforms[i, 2]:transforms[i, 3]] = 1
    masks = masks.reshape(N, 2, -1)
    return no_mask * masks


def get_box_coordinates_matrix1(shift_H, shift_W, height, width):
    ind = torch.tensor(np.indices((height, width))).float().reshape(2, -1)
    ind[0, :] += shift_H
    ind[1, :] += shift_W
    return ind

def to_pil(image):
    return (image * 255).int().permute(1, 2, 0).detach().numpy()