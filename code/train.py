import torch


class Trainer:
    def __init__(self, device):
        self.device = device

    def train(self, model, criterion, optimizer, train_loader, test_loader, num_epochs):
        for epoch in range(num_epochs):
            loss, val_loss = 0, 0
            for iteration, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                data = model(batch)
                curr_loss = criterion(data, batch)
                print(curr_loss)
                loss += curr_loss.item()
                curr_loss.backward()
                optimizer.step()
                if iteration % 1 == 0:
                    print(f"iteration is {iteration}, curr_loss is ${curr_loss}, total loss is ${loss / iteration}")

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    data = model(batch)
                    curr_loss = criterion(data, batch)
                    val_loss += curr_loss

            mean_loss = loss / len(train_loader)
            mean_val_loss = val_loss / len(test_loader)
            print("After epoch {} loss is {} and validation loss is {}".format(epoch, mean_loss, mean_val_loss))
