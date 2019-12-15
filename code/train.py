import torch

import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, device):
        self.device = device

    def train(self, model, criterion, optimizer, train_loader, test_loader, num_epochs):
        for epoch in range(num_epochs):
            loss, val_loss = 0, 0
            for iteration, (x, y) in enumerate(train_loader):
                print('y:', y.shape)
                optimizer.zero_grad()
                output = model(*x)
                plt.imsave(f'./training/{epoch}_{iteration}.png', output[0].detach().numpy(), cmap='gray', vmin=0, vmax=1)
                curr_loss = criterion(output, y)
                loss += curr_loss.item()
                curr_loss.backward()
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
