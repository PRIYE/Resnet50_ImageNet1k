import lightning as L
import torch
import torch.nn.functional as F
import timm
from torch import optim
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


class ImageNetClassifier(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        #self.model = timm.create_model('resnet18', pretrained=True, num_classes=10)
        self.model = timm.create_model('resnet50', pretrained=False, num_classes=1000)
        self.train_acc = Accuracy(task="multiclass", num_classes=1000)
        self.val_acc = Accuracy(task="multiclass", num_classes=1000)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # # Calculate total steps
        # total_steps = self.trainer.estimated_stepping_batches

        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr,
        #     total_steps=total_steps,
        #     pct_start=0.3,
        #     div_factor=25,
        #     final_div_factor=1e4,
        #     three_phase=False,
        #     anneal_strategy='cos'
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step"  # Update at every step
        #     }
        # }
        # Define ReduceLROnPlateau Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.1, patience=2, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" , 
                'frequency': 1,
                'monitor': 'val_loss',
            }
        }
