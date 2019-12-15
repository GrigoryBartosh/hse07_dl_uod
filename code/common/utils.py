import torch.nn as nn;


def activation_by_name(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'selu':
        return nn.SELU(inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'none':
        return None
    else:
        assert False, f"Unsupported activation: {activation}"