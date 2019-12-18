import torch
import numpy as np
import torch.nn
import os

from PIL import Image


def load_sample(number):
    # return Image.open(os.path.join('./../datasets/emoji', f'test_{number}.png'))
    return Image.open(os.path.join('./../datasets/black', 'black.png'))


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
        X = self.filter_sugggest(X)

        result = torch.zeros((n_pictures,) + self.image_shape)
        for batch_number in range(n_pictures):
            for suggest in range(X.shape[1]):
                data = X[batch_number][suggest]
                image = self.get_image(data[5:].argmax())
                x1 = torch.round(data[0] * self.image_shape[1]).int()
                x2 = torch.round(data[1] * self.image_shape[1]).int()
                y1 = torch.round(data[2] * self.image_shape[2]).int()
                y2 = torch.round(data[3] * self.image_shape[2]).int()
                if x1 < 0 or y1 < 0 or x2 > self.image_shape[1] or y2 > self.image_shape[2] or x2 <= x1 or y2 <= y1:
                    continue
                result[batch_number] += apply_transform(image, torch.tensor([x1, x2, y1, y2]))

        return result[:, :3]

        # img = torch.ones(n_pictures, *self.image_shape, dtype=torch.float32)
        #
        # for image, data in zip(img, X):  # TODO(anyone): rewrite in vector terms?
        #     x = torch.round(data[:, 0] * self.image_shape[1]).int()
        #     y = torch.round(data[:, 1] * self.image_shape[2]).int()
        #     h = torch.round(data[:, 2] * self.image_shape[1]).int()
        #     w = torch.round(data[:, 3] * self.image_shape[2]).int()
        # h, w = torch.full(h.shape, 64).int(), torch.full(w.shape, 64).int()
        # d = data[:, 4]
        # c = data[:, 5:]

        # order = np.arange(n_emoji)[torch.argsort(d, dim=1)].T
        # for i in np.arange(n_emoji)[torch.argsort(d)].reshape(-1):
        #     x1, y1, x2, y2 = x[i] - h[i] // 2, y[i] - w[i] // 2, x[i] + (h[i] + 1) // 2, y[i] + (w[i] + 1) // 2
        #     if x1 < 0 or y1 < 0 or x2 > self.image_shape[1] or y2 > self.image_shape[2]:
        #         continue
        #
        #     new_image = self.get_image(c[i].argmax())
        #     c_new = new_image[:3, :, :]
        #     alpha_new = new_image[3, :, :].repeat(3, 1, 1)
        #     c_old = image[:3, x1:x2, y1:y2]
        #     alpha_old = image[3, x1:x2, y1:y2].repeat(3, 1, 1)
        #     alpha_0 = alpha_new + alpha_old * (1 - alpha_new)
        #
        #     image[:3, x1:x2, y1:y2] = (c_new * alpha_new + c_old * alpha_old * (1 - alpha_new)) / alpha_0
        #     image[3, x1:x2, y1:y2] = alpha_0[0, :, :]
        #
        # return img[:, :3]

    def filter_sugggest(self, X):
        return X[:, :1]

    def get_images(self, X):
        return X[:, :, :4]

    def get_images(self, X):
        X = X[:, :, 5:].argmax(dim=2)
        # TODO
        return X


def get_transform_matrix(row_shift, col_shift, row_scale, col_scale):
    return torch.tensor([
        [row_scale, 0, row_shift],
        [0, col_scale, col_shift],
        [0, 0, 1]
    ]).float()


def get_coordinates_matrix(size):
    return torch.tensor(np.vstack([
        np.indices((size, size)).reshape(2, -1),
        np.ones(size * size)
    ])).float()


def get_box_coordinates_matrix(shift_H, shift_W, height, width):
    ind = torch.tensor(np.indices((height, width))).float().reshape(2, -1)
    ind[0, :] += shift_H
    ind[1, :] += shift_W
    return ind


# image : image_size x image_size
# transform : row1, row2, col1, col2 -- coordinates [0 - image_size)
def apply_transform(image, transform):
    CH, image_row, image_col = image.shape
    assert (image_col == image_row)
    image_size = image_row
    image_WH = image_col * image_row
    row1, row2, col1, col2 = transform
    box_row = row2 - row1
    box_col = col2 - col1
    box_WH = box_row * box_col
    T = get_transform_matrix(row1 * 1.0, col1 * 1.0, box_row * 1.0 / image_size, box_col * 1.0 / image_size)
    print("T", T)
    C = get_coordinates_matrix(image_size)
    TC = T @ C
    TC = TC[:2]
    TC = TC.repeat(box_WH, 1, 1)  # box x 2 x image_WH
    B = get_box_coordinates_matrix(row1, col1, box_row, box_col)
    B = B.repeat(image_WH, 1, 1).permute(2, 1, 0)  # box x 2 x WH
    EPS = 1e-8
    #     A = (TC - B).pow(2).sum(1) + EPS
    #     A = (A.T / A.max(dim=1)[0].T).T
    #     A = ((1 - A) / A).log()
    A = (TC - B).pow(2).sum(1)
    A = -A
    A = A.softmax(dim=1)
    P = image.reshape(CH, -1).permute(1, 0)
    R = (A @ P).reshape(box_row, box_col, CH).permute(2, 0, 1)
    RES = torch.ones(CH, image_row, image_col)
    #     real_rows_end = row2 - row1 if row2 < image_row else image_row - row1
    #     real_cols_end = col2 - col1 if col2 < image_col else image_col - col1
    #     real_rows_beg = row2 - row1 if row2 < image_row else image_row - row1
    #     real_cols_beg = col2 - col1 if col2 < image_col else image_col - col1
    #     RES[:, row1:row2, col1:col2] = R[:, real_rows_beg:real_rows_end,  real_cols_beg:real_cols_end]
    RES[:, row1:row2, col1:col2] = R

    return RES
