import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


class Trainer:
    def __init__(self, device):
        self.device = device

    def train(self, model, criterion, optimizer, train_loader, test_loader, num_epochs):
        model = model.to(self.device)
        for epoch in range(num_epochs):
            loss, val_loss = 0, 0
            for iteration, x in enumerate(train_loader):
                optimizer.zero_grad()
                with torch.autograd.detect_anomaly():
                    x = x.to(self.device)
                    output = model(x)
                    print("out", output.shape)
                    print("in", x.shape)
                    image = (output[0] * 255).detach().cpu().numpy().transpose(1, 2, 0)
                    # image[image[:, :, 3] <= 11] = 255
                    # image = image[:,:,:3]

                    plt.imsave(f"test_{iteration}.png", image.astype('uint8'))
                    small_x = torch.zeros((x.shape[0], 3, 100, 100))
                    for ind in range(len(x)):
                        pil = Image.fromarray(x[ind].detach().int().cpu().numpy().transpose(1, 2, 0).astype('uint8'))
                        resized = np.asarray(pil.resize((100, 100)))
                        small_x[ind] = torch.tensor(resized.transpose(2, 0, 1))

                    curr_loss = criterion(output, small_x.to(self.device))
                    curr_loss.backward()
                    loss += curr_loss.item()
                    optimizer.step()
                    if iteration % 1 == 0:
                        print(f"iteration is {iteration}, curr_loss is ${curr_loss}, total loss is ${loss / (iteration + 1)}")

            # with torch.no_grad():
            #     for (x, y) in test_loader:
                    # x = x.to(self.device)
                    # output = model(*x)
                    # curr_loss = criterion(output, y)
                    # val_loss += curr_loss

            mean_loss = loss / len(train_loader)
            mean_val_loss = val_loss / len(test_loader)
            print("After epoch {} loss is {} and validation loss is {}".format(epoch, mean_loss, mean_val_loss))
