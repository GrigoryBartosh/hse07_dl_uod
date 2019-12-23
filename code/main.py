from code.pipeline import Pipeline
from code.train import *


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    device = get_device()
    pipeline = Pipeline(device)
    trainer = Trainer(device)
    torch.autograd.set_detect_anomaly(True)
    trainer.train(
        pipeline.model,
        pipeline.criterion,
        pipeline.optimizer,
        pipeline.train_loader,
        pipeline.test_loader,
        pipeline.num_epochs
    )
