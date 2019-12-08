from code.train import *


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    device = get_device()
    trainer = Trainer(device)
    # trainer.train()
    # print('hello {}'.format('there'))
