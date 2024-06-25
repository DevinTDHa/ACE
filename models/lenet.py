import torch
from torch import nn
import lightning as L


class LeNet5(L.LightningModule):
    def __init__(self, num_classes=1, loss=nn.MSELoss(), learing_rate=0.001):
        super(LeNet5, self).__init__()
        num_channels = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # output is 16 x 13 x 13
        self.fc = nn.Linear(16 * 13 * 13, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

        # Hyperparameters
        self.loss = loss
        self.learning_rate = learing_rate

        self.save_hyperparameters()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict({"loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict({"val_loss": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()
        self.log_dict({"lr": sch.get_last_lr()[0]}, prog_bar=True)

def load_lenet(path):
    model = LeNet5.load_from_checkpoint(path)
    return model