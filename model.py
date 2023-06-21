import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
import numpy as np
from tqdm import tqdm
from augmentation.mixup import mixup, aug_criterion
from augmentation.cutmix import cutmix
from augmentation.cutout import Cutout
from augmentation.fmix import fmix

class Module(pl.LightningModule):
    def __init__(
        self, 
        model_name='tf_efficientnet_b0_ns', 
        image_size=512, 
        num_classes=10, 
        lr=1e-3, 
        max_lr=1e-2, 
        num_epochs=10, 
        steps_per_epoch=100, 
        weight_decay=1e-6, 
        aug=None, 
        p=0,
        pretrained=True
    ):
        super(Module, self).__init__()

        # init layer in __init__ will auto set device by lightning
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.multiple_dropout = [nn.Dropout(0.25) for i in range(8)]
        self.fc = nn.Linear(in_features * 2, 10)
        self.pooled_features_avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pooled_features_max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.image_size = image_size
        self.num_classes = num_classes
        self.lr = lr
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.weight_decay = weight_decay
        self.aug = aug
        self.p = p
        print(f'Using {self.aug} augmentation \n')

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=self.num_classes, task="multiclass")

    def forward(self, images: torch.Tensor):
        features = self.model(images)
        pooled_features = torch.cat([self.pooled_features_avg(features), 
                                    self.pooled_features_max(features)], 
                                    dim=1).flatten(1)
        pooled_features_dropout = torch.zeros((pooled_features.shape), device=self.device)
        for i in range(8):
            pooled_features_dropout += self.multiple_dropout[i](pooled_features)
        pooled_features_dropout /= 8
        outputs = self.fc(pooled_features_dropout)
        return outputs

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                            epochs=self.num_epochs,
                                                            steps_per_epoch=self.steps_per_epoch,
                                                            max_lr=self.max_lr,
                                                            pct_start=0.2,
                                                            div_factor=1.0e+3,
                                                            final_div_factor=1.0e+3)
        scheduler = {'scheduler': self.scheduler, 'interval': 'step',}
        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target']

        if self.aug is not None:

            if self.aug == 'cutout':

                output = self(image)

                loss = self.criterion(output, target)
                
            else:

                decision = np.random.rand()

                if decision > 0 and decision <= self.p:
                
                    shuffled_target, lam = 0, 0

                    if self.aug == 'mixup':
                        image, target, shuffled_target, lam = mixup(image, target, alpha=1.0, device=self.device)

                    elif self.aug == 'cutmix':
                        image, target, shuffled_target, lam = cutmix(image, target, alpha=1.0, device=self.device)

                    elif self.aug == 'fmix':
                        image, target, shuffled_target, lam = fmix(image, target, alpha=1.0, decay_power=3, shape=(self.image_size, self.image_size), device=self.device)
                    
                    output = self(image)

                    loss = lam*self.criterion(output, target) + (1-lam)*self.criterion(output, shuffled_target)

                else:

                    output = self(image)

                    loss = self.criterion(output, target)

        else:
            
            output = self(image)

            loss = self.criterion(output, target)

        accuracy = self.accuracy(output.argmax(1), target)

        logs = {'train_loss': loss, 'train_acc': accuracy, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target']

        output = self(image)

        loss = self.criterion(output, target)

        accuracy = self.accuracy(output.argmax(1), target)

        logs = {'valid_loss': loss, 'valid_acc': accuracy}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path, device):
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module