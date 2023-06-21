import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
from data_module import DataModule
from model import Module

def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = ArgumentParser()
home_dir = '/content'
parser.add_argument('--train-dir', default=f'{home_dir}/data', type=str)
parser.add_argument('--test-dir', default=f'{home_dir}/data', type=str)
parser.add_argument('--model-name', default='tf_efficientnet_b0_ns', type=str)
parser.add_argument('--num-classes', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--max-lr', default=1e-2, type=float)
parser.add_argument('--num-epochs', default=10, type=int)
parser.add_argument('--steps-per-epoch', default=100, type=int)
parser.add_argument('--weight-decay', default=1e-6, type=float)
parser.add_argument('--precision', default='16-mixed', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--aug', default=None, type=none_or_str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-workers', default=2, type=int)
parser.add_argument('--n-splits', default=5, type=int)
parser.add_argument('--image-size', default=512, type=int)
parser.add_argument('--n-valid', default=0, type=int)
parser.add_argument('--p', default=0, type=float)
parser.add_argument('--project', default=None, type=none_or_str)
parser.add_argument('--name', default=None, type=none_or_str)
parser.add_argument('--id', default=None, type=none_or_str)
parser.add_argument('--resume', default=None, type=none_or_str, help='path/to/checkpoint.pt')
parser.add_argument('--cpkt_dir', default=None, type=none_or_str)

args = parser.parse_args()

def train(args):
    pl.seed_everything(args.seed)

    if args.aug is not None:
        print(f'p for augmentation: {args.p}')

    datamodule = DataModule(train_dir=args.train_dir, test_dir=args.test_dir,
                            n_valid=args.n_valid, n_splits=args.n_splits, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers,
                            aug=args.aug, p=args.p)
    datamodule.setup()
    args.steps_per_epoch = len(datamodule.train_dataloader())

    module = Module(model_name=args.model_name, 
                    num_classes=args.num_classes, 
                    lr=args.lr, 
                    max_lr=args.max_lr, 
                    num_epochs=args.num_epochs, 
                    steps_per_epoch=args.steps_per_epoch,
                    weight_decay=args.weight_decay, 
                    aug=args.aug, 
                    p=args.p,
                    pretrained=args.pretrained)

    wandb_logger = WandbLogger(project=args.project, name=args.name, id=args.id)

    best_val_loss_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.cpkt_dir,
        monitor='valid_loss',
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        filename='{best_loss}-{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
        verbose=False,
        mode='min'
    )

    best_val_acc_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.cpkt_dir,
        monitor='valid_acc',
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        filename='{best_acc}-{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        devices="auto",
        accumulate_grad_batches=1,
        precision=args.precision,
        callbacks=[best_val_loss_checkpoint_callback, best_val_acc_checkpoint_callback],
        logger=wandb_logger
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path = args.resume)

    return trainer

if __name__ == '__main__':
    train(args)