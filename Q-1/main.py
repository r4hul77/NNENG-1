import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import cv2
import torchvision.transforms as transforms
import os
from dataset import Image_Dataset
from pytorch_lightning.callbacks import Callback
import wandb
import torchmetrics

# Function which takes in the string and returns the first word of the string
# input string format : LabelNumber.jpg
# output string format : Label


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                           for x, pred, y in zip(val_imgs[:self.num_samples],
                                                 preds[:self.num_samples],
                                                 val_labels[:self.num_samples])]
            })


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.back_bone = torchvision.models.resnet18(pretrained=True)

        # freeze the backbone
        for param in self.back_bone.parameters():
            param.requires_grad = False
        num_filters = self.back_bone.fc.in_features
        self.back_bone.fc = nn.Linear(num_filters, num_classes)

        self.accuracy = torchmetrics.Accuracy()
        self.loss = nn.CrossEntropyLoss()

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size


    # will be used during inference
    def forward(self, x):
        x = self.back_bone(x)
        x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)


        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# This class loads the data from weatehrDataset to feed into pytorch-lighting module

def collate_fn(batch):
    return torch.stack([item[0] for item in batch]), torch.Tensor([item[1] for item in batch]).long()

if __name__ == '__main__':
    input_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_Dataset = Image_Dataset(image_base_dir='weatherDataset', transform=input_transforms)
    image_Dataset.init()
    train_dataset, val_dataset, test_dataset = image_Dataset.split_train_val_test(0.2, 0.2)
    model = LitModel(num_classes=3)
    train_dataloader, val_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn),\
                                                        DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn),\
                                                        DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    wandb_logger = WandbLogger(project='weather_classification', name='resnet18')
    wandb_logger.watch(model)
    trainer = pl.Trainer(gpus=1, max_epochs=10, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)
