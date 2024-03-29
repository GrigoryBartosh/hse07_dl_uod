import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

__all__ = ['Stacker']

import torchvision

from torch.utils.data import DataLoader

from code.common.utils import activation_by_name
from code.dataset import EmojiDataset, MovableDataset
from code.train import Trainer


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1,
        stride=stride, padding=0, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=7,
        stride=stride, padding=3, bias=False
    )


def upsample3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=stride),
        conv3x3(in_planes, out_planes)
    )


def upsample7x7(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=stride),
        conv7x7(in_planes, out_planes)
    )


class SimpleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes=None, stride=1,
                 activ='relu', upsample_block=False):
        super(SimpleBlock, self).__init__()

        if out_planes is None:
            out_planes = in_planes * self.expansion

        conv = upsample3x3 if upsample_block else conv3x3

        self.conv1 = conv(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.activ1 = activation_by_name(activ)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ1(out)

        return out


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes=None, out_planes=None, stride=1,
                 activ='relu', upsample_block=False):
        super(ResBasicBlock, self).__init__()

        if planes is None:
            planes = in_planes // self.expansion

        if out_planes is None:
            out_planes = planes * self.expansion

        self.residual = None
        if stride != 1 or in_planes != out_planes:
            self.residual = nn.Sequential(
                (upsample3x3 if upsample_block else conv1x1)(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
            )

        conv = upsample3x3 if upsample_block else conv3x3

        self.activ = activation_by_name(activ)

        self.conv1 = conv(in_planes, planes, 1 if upsample_block else stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, out_planes, stride if upsample_block else 1)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual is not None:
            identity = self.residual(x)

        out += identity
        out = self.activ(out)

        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes=None, out_planes=None, stride=1,
                 activ='relu', upsample_block=False):
        super(ResBottleneck, self).__init__()

        if planes is None:
            planes = in_planes // self.expansion

        if out_planes is None:
            out_planes = planes * self.expansion

        self.residual = None
        if stride != 1 or in_planes != out_planes:
            self.residual = nn.Sequential(
                (upsample3x3 if upsample_block else conv1x1)(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
            )

        conv = upsample3x3 if upsample_block else conv3x3

        self.activ = activation_by_name(activ)

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activ(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.residual is not None:
            identity = self.residual(x)

        out += identity
        out = self.activ(out)

        return out


class Encoder(nn.Module):
    def __init__(self, block, layer_sizes, activ='relu'):
        super(Encoder, self).__init__()

        in_planes = 64
        self.conv1 = conv7x7(1, in_planes, stride=2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.activ1 = activation_by_name(activ)
        self.conv2 = conv3x3(in_planes, in_planes, stride=2)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.activ2 = activation_by_name(activ)

        layers = [self._make_layer(block, in_planes, in_planes, 1, layer_sizes[0], activ)]
        for layer_size in layer_sizes[1:]:
            layers += [self._make_layer(block, in_planes, in_planes * 2, 2, layer_size, activ)]
            in_planes = in_planes * 2

        self.layer = nn.Sequential(*layers)

    def _make_layer(self, block, in_planes, out_planes, stride, layer_size, activ):
        layers = [block(
            in_planes, out_planes=out_planes,
            stride=stride, activ=activ
        )]

        for _ in range(1, layer_size):
            layers += [block(out_planes, activ=activ)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activ2(x)

        x = self.layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, block, layer_sizes, in_planes=512, start_planes=512, activ='lrelu'):
        super(Decoder, self).__init__()

        layers = [self._make_layer(block, in_planes, start_planes // 2, 2, layer_sizes[0], activ)]
        in_planes = start_planes // 2
        for layer_size in layer_sizes[1:-1]:
            layers += [self._make_layer(block, in_planes, in_planes // 2, 2, layer_size, activ)]
            in_planes = in_planes // 2
        layers += [self._make_layer(block, in_planes, in_planes, 1, layer_sizes[-1], activ)]
        self.layer = nn.Sequential(*layers)

        self.conv1 = upsample3x3(in_planes, out_planes=64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.activ1 = activation_by_name(activ)
        self.conv2 = upsample7x7(in_planes=64, out_planes=1, stride=2)
        self.activ2 = nn.Tanh()

    def _make_layer(self, block, in_planes, out_planes, stride, layer_size, activ):
        layers = []
        for _ in range(layer_size - 1):
            layers += [block(in_planes, activ=activ, upsample_block=True)]

        layers += [block(
            in_planes=in_planes, out_planes=out_planes,
            stride=stride, activ=activ, upsample_block=True
        )]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ1(x)

        x = self.conv2(x)
        x = self.activ2(x)

        return x


class Mover(nn.Module):
    def __init__(self, args):
        super(Mover, self).__init__()

        layer_sizes = args['encoder']['layers']
        self.encoder = Encoder(
            self.get_block_by_name(args['encoder']['block']),
            layer_sizes,
            args['encoder']['activ']
        )

        planes = 64 * 2 ** (len(layer_sizes) - 1)

        self.decoder = Decoder(
            self.get_block_by_name(args['decoder']['block']),
            args['decoder']['layers'],
            planes + args['params_move_count'],
            planes,
            args['decoder']['activ']
        )

    def get_block_by_name(self, block_name):
        if block_name == 'SimpleBlock':
            return SimpleBlock
        elif block_name == 'ResBasicBlock':
            return ResBasicBlock
        elif block_name == 'ResBottleneck':
            return ResBottleneck

        assert False, f"Unsupported block: {self.args['block']}"

    def forward(self, x, x_params_move):
        print('x:', x.shape)
        print('x_params:', x_params_move.shape)
        x = x[:, None, :, :]

        x = self.encoder(x)

        x_params_move = x_params_move[:, :, None, None]
        x_params_move = x_params_move.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat((x, x_params_move), 1)

        x = self.decoder(x)

        x = x.squeeze(1)
        print('out_inside:', x.shape)

        return x


class Stacker(nn.Module):
    def __init__(self, args):
        super(Stacker, self).__init__()

        self.mover = Mover(args)

    def move(self, x_t, x_params_move):
        return self.mover(x_t, x_params_move)

    def stack(self, x_i, x_t, x_rgb):
        x_t = x_t[:, None, :, :]
        x_rgb = x_rgb[:, :, None, None]
        return x_i * (1 - x_t) + x_rgb * x_t

    def forward(self, x_i, x_t, x_params):
        x_params_move, x_rgb = x_params[:, :5], x_params[:, 5:]
        x_t = self.move(x_t, x_params_move)
        return self.stack(x_i, x_t, x_rgb)

def get_stacker():
    params_move_count = 5
    stacker_model = {
        'encoder': {
            'block': 'ResBasicBlock',
            'layers': [2, 2, 2, 2],
            'activ': 'relu'
        },
        'decoder': {
            'block': 'ResBasicBlock',
            'layers': [2, 2, 2, 2],
            'activ': 'lrelu'
        },
        'params_move_count': params_move_count
    }
    return Stacker(stacker_model)


def get_mover():
    params_move_count = 5
    stacker_model = {
        'encoder': {
            'block': 'ResBasicBlock',
            'layers': [2, 2, 2, 2],
            'activ': 'relu'
        },
        'decoder': {
            'block': 'ResBasicBlock',
            'layers': [2, 2, 2, 2],
            'activ': 'lrelu'
        },
        'params_move_count': params_move_count
    }
    return Mover(stacker_model)

class ReceptieveFieldMseLoss:

    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, data, expected, box):
        N, W, H = data.shape
        x = torch.round(box[:, 0] * W).int()
        y = torch.round(box[:, 1] * H).int()
        h = torch.round(box[:, 2] * W).int()
        w = torch.round(box[:, 3] * H).int()
        loss = 0
        for i in range(N):
            x1, y1, x2, y2 = x[i] - h[i] // 2, y[i] - w[i] // 2, x[i] + (h[i] + 1) // 2, y[i] + (w[i] + 1) // 2
            loss += self.mse_loss(data[:, x1:x2, y1:y2], expected[:, x1:x2, y1:y2])
        return loss / N


if __name__ == '__main__':
    # image = torch.zeros((64, 64)).float()
    # plt.imsave('./../../datasets/black/black.png', image.detach().numpy(), cmap='gray', vmin=0, vmax=1)
    # exit(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    trainer = Trainer(device)
    # image = torch.zeros(2, 80, 100)
    # move_params = torch.ones(2, 5)
    model = get_mover()
    # res = model(image, move_params)
    # print('-----------')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    batch_size = 32
    train_data = MovableDataset(
        root=os.path.join('./../../datasets/mover/', 'train'),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ])
    )
    test_data = MovableDataset(
        root=os.path.join('./../../datasets/mover/', 'test'),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ])
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    # for (x, moves), y in train_loader:
        # plt.imsave('./x.png', torch.ones(y[0].shape).numpy(), cmap='gray', vmin=0, vmax=1)
        # print(moves[0])
        # plt.imsave('./y.png', y[0].numpy(), cmap='gray', vmin=0, vmax=1)
        # loss = ReceptieveFieldMseLoss()
        # print(loss(y[:1], torch.ones(y.shape)[:1], moves[:1]))
        # break
    trainer.train(model.to(device), ReceptieveFieldMseLoss(), optimizer, train_loader, test_loader, 20, device)
    # print(res.shape)
