import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        degrad_patch, clean_patch = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=300,
        )

        return [optimizer], [scheduler]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        degrad_patch, clean_patch = batch
        restored = self.net(degrad_patch)
        mse = F.mse_loss(restored, clean_patch)
        psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
        self.log("val_psnr", psnr, prog_bar=True)


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt, part='train')
    valset = PromptTrainDataset(opt, part='val')
    valloader = DataLoader(
        valset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        every_n_epochs=1,
        save_top_k=-1
    )

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    model = PromptIRModel()

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader
    )


if __name__ == '__main__':
    main()
