import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
import torch

from datamodules.imagenet_datamodule import ImageNetDataModule
from models.classifier import ImageNetClassifier

class NewLineProgressBar(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\nEpoch {trainer.current_epoch}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', 0)
        train_acc = metrics.get('train_acc', 0)
        print(f"\rTraining - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}", end="")
    
    def on_validation_epoch_start(self, trainer, pl_module):
        print("\n\nValidation:")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss', 0)
        val_acc = metrics.get('val_acc', 0)
        print(f"\rValidation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}", end="")

def find_optimal_lr(model, data_module):
    # Initialize LRFinder
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    # Run LR finder with stage parameter
    data_module.setup(stage='fit')
    lr_finder.range_test(data_module.train_dataloader(), end_lr=1, num_iter=200, step_mode="exp")
    
    # Get the learning rate with the steepest gradient
    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']
    
    # Find the learning rate with minimum loss
    optimal_lr = lrs[losses.index(min(losses))]
    
    # You might want to pick a learning rate slightly lower than the minimum
    optimal_lr = optimal_lr * 0.1  # Common practice to use 1/10th of the value
    
    print(f"Optimal learning rate: {optimal_lr}")
    
    # Plot the LR finder results
    lr_finder.plot()  # Will save the plot
    lr_finder.reset()  # Reset the model and optimizer
    
    return optimal_lr

def main(chkpoint_path=None):
    if chkpoint_path is not None:
        model = ImageNetClassifier(lr=1e-2)
        data_module = ImageNetDataModule(batch_size=256, num_workers=8)
        checkpoint_callback = ModelCheckpoint(
            dirpath="logs/checkpoints",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3
        )

        # Initialize Trainer
        trainer = L.Trainer(resume_from_checkpoint=chkpoint_path,
            max_epochs=epochs,
            precision="bf16-mixed",
            callbacks=[
                checkpoint_callback,
                NewLineProgressBar(),
                TQDMProgressBar(refresh_rate=1)
            ],
            accelerator="auto",
            logger=TensorBoardLogger(save_dir="logs", name="image_net_classifications"),
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=1,
            val_check_interval=1.0,
            check_val_every_n_epoch=1
        )
        trainer.fit(model, data_module)
    else:
        # Create directories
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        # Initialize DataModule and Model
        data_module = ImageNetDataModule(batch_size=256, num_workers=8)
        model = ImageNetClassifier(lr=1e-2)  # Initial lr will be overridden

        # Find optimal learning rate
        optimal_lr = find_optimal_lr(model, data_module)
        #optimal_lr = 6.28E-02
        # Calculate total steps for OneCycleLR
        epochs = 60
        data_module.setup(stage='fit')
        steps_per_epoch = len(data_module.train_dataloader())
        total_steps = epochs * steps_per_epoch

        # # Initialize optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=optimal_lr)

        # # Initialize OneCycleLR scheduler
        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=optimal_lr,
        #     total_steps=total_steps,
        #     pct_start=0.3,  # Spend 30% of time increasing LR
        #     div_factor=25,  # Initial LR will be max_lr/25
        #     final_div_factor=1e4,  # Final LR will be max_lr/10000
        #     three_phase=False,  # Use one cycle policy
        #     anneal_strategy='cos'  # Use cosine annealing
        # )
        model = ImageNetClassifier(lr=optimal_lr)  # Initial lr will be overridden
        # Initialize callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath="logs/checkpoints",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3
        )

        # Initialize Trainer
        trainer = L.Trainer(
            max_epochs=epochs,
            precision="bf16-mixed",
            callbacks=[
                checkpoint_callback,
                NewLineProgressBar(),
                TQDMProgressBar(refresh_rate=1)
            ],
            accelerator="auto",
            logger=TensorBoardLogger(save_dir="logs", name="image_net_classifications"),
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=1,
            val_check_interval=1.0,
            check_val_every_n_epoch=1
        )

        # Train the model
        trainer.fit(model, data_module)

if __name__ == "__main__":
    main(chkpoint_path=None)
