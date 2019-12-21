import shutil

import torch
import numpy as np
import torch.nn
import os
import argparse
import matplotlib.pyplot as plt

from code.model.decoder import Decoder


def load_generated(n_samples, path):
    data = []
    for i in range(n_samples):
        data.append([])
        with open(os.path.join(path, f'{i}.txt'), 'r') as in_file:
            for line in in_file.readlines()[1:]:
                data[-1].append(torch.tensor(list(map(np.float32, line.split(' ')))))
    return data


class Generator(object):
    def __init__(self, target_shape=(4, 300, 300)):
        self.target_shape = target_shape
        self.decoder = Decoder(image_shape=target_shape)
        self._gen_params = None

    def gen_image(self, n_pictures=5):
        depths = np.random.uniform(0, 1, n_pictures)

        output = []
        for i in range(n_pictures):
            classes = np.random.uniform(0, 1, self.decoder.n_images)
            classes = classes / classes.sum()
            image_shape = self.decoder.get_image(classes.argmax()).shape
            x = np.random.uniform((image_shape[1] // 2) / self.target_shape[1],
                                  1 - ((1 + image_shape[1]) // 2 + 1) / self.target_shape[1])
            y = np.random.uniform((image_shape[2] // 2) / self.target_shape[2],
                                  1 - ((1 + image_shape[2]) // 2 + 1) / self.target_shape[2])
            output.append([x,
                           y,
                           image_shape[1] / self.target_shape[1],
                           image_shape[2] / self.target_shape[2],
                           depths[i],
                           *classes])

        self._gen_params = output
        return self.decoder.forward1(torch.from_numpy(np.array(self._gen_params))[None])[0]

    def __call__(self, n_samples, output_dir, n_emoji_range=(5, 6), test_train_split=0.8):
        self.n_samples = n_samples
        self.output_dir = output_dir
        self._generate_to_folder(n_samples * test_train_split,
                                 os.path.join(output_dir, 'train/data'),
                                 os.path.join(output_dir, 'train/info'),
                                 n_emoji_range)

        self._generate_to_folder(n_samples * (1 - test_train_split),
                                 os.path.join(output_dir, 'test/data'),
                                 os.path.join(output_dir, 'test/info'),
                                 n_emoji_range)

    def _generate_to_folder(self, n_samples, output_dir, info_dir, n_emoji_range):
        for directory in [output_dir, info_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        for ind in range(int(n_samples)):
            n_pictures = np.random.randint(n_emoji_range[0], n_emoji_range[1])
            image = self.gen_image(n_pictures).detach().numpy()
            image_path = os.path.join(output_dir, f'{ind}.png')
            plt.imsave(image_path, image.transpose(1, 2, 0))
            self.save_description(os.path.join(info_dir, f'{ind}.txt'))

    def save_description(self, path):
        with open(path, 'w') as output_file:
            output_file.write(
                f'# 0 <= x <= 1\t0 <= y <= 1\t0 <= h <= 1\t0 <= w <= 1\t0 <= d <= 1\t0 <= probs <= 1{os.linesep}')
            for params in self._gen_params:
                output_file.write(' '.join(map(str, params)))
                output_file.write(os.linesep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='./../datasets/mover', type=str, help='Path to save generated dataset')
    parser.add_argument('--n_samples', default=200, type=int, help='Number of samples to generate')
    parser.add_argument('--n_min_pictures', default=1, type=int, help='Min number of emoji on one picture')
    parser.add_argument('--n_max_pictures', default=2, type=int, help='Max number of emoji on one picture')
    args = parser.parse_args()

    gen = Generator()
    gen(output_dir=args.output_path, n_emoji_range=(args.n_min_pictures, max(args.n_max_pictures,
                                                                             args.n_min_pictures + 1)),
        n_samples=args.n_samples)
