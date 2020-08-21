import logging
import seotbx
import numpy as np
import seotbx.polsarproc.definitions as defs
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import copy

logger = logging.getLogger("seotbx.polsarproc.dll.apps.train")
from torch.utils.data import random_split, DataLoader
import torchvision


def train_parser_func(subparsers, mode, argparse=None):
    ap = subparsers.add_parser(mode, help="Basic MISC application")
    ap.add_argument("save_dir", type=str, help="path to the session output root directory")
    ap.add_argument('--in_features', default=28 * 28, type=int)
    ap.add_argument('--hidden_dim', default=50000, type=int)
    # use 500 for CPU, 50000 for GPU to see speed difference
    ap.add_argument('--out_features', default=10, type=int)
    ap.add_argument('--drop_prob', default=0.2, type=float)

    # data
    ap.add_argument('--data_root', default="", type=str)
    ap.add_argument('--num_workers', default=4, type=int)

    # training params (opt)
    ap.add_argument('--epochs', default=20, type=int)
    ap.add_argument('--batch_size', default=64, type=int)
    ap.add_argument('--learning_rate', default=0.001, type=float)
    ap.add_argument('--gpus', default=1, type=int)


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core import LightningModule, LightningDataModule

seed_everything(234)

if 0:
    class BaseData(LightningDataModule):

        def __init__(self, config, **kwargs):
            # init superclass
            super().__init__()

            obj_conf = seotbx.utils.get_key("datasets", config)
            datasets_keys = seotbx.utils.get_key("datasets_keys", obj_conf)
            self.datasets = {}

            for dataset_key in ["train_dataset", "val_dataset", "test_dataset"]:

                if dataset_key == "test_dataset":
                    continue

                dataset_id = datasets_keys[dataset_key]
                ds_config = seotbx.utils.get_key(dataset_id, obj_conf)
                obj_type = seotbx.utils.get_key("type", ds_config)
                obj_params = seotbx.utils.get_key("params", ds_config)

                transforms_configs = seotbx.utils.get_key_def("transform", obj_params, default=[])
                transforms = []
                for transform_conf in transforms_configs:
                    tr_obj_type = seotbx.utils.get_key("type", transform_conf)
                    tr_obj_params = seotbx.utils.get_key_def("params", transform_conf, default={})
                    tr_obj_class = seotbx.utils.import_class(tr_obj_type)
                    transforms.append(tr_obj_class(**tr_obj_params))

                obj_params["transform"] = torchvision.transforms.Compose(transforms)

                obj_class = seotbx.utils.import_class(obj_type)
                self.datasets[dataset_key] = obj_class(**obj_params)

            def train_dataloader(self):
                return DataLoader(self.datasets["train_dataset"], batch_size=16, num_workers=4)

            def val_dataloader(self):
                return DataLoader(self.datasets["val_dataset"], batch_size=16, num_workers=4)


class BaseModel(LightningModule):
    def __init__(self, config, **kwargs):
        # init superclass
        super().__init__()

        model_config = seotbx.utils.get_key("model", config)

        obj_conf = seotbx.utils.get_key("nn", model_config)
        obj_type = seotbx.utils.get_key("type", obj_conf)
        obj_class = seotbx.utils.import_class(obj_type)
        obj_params = seotbx.utils.get_key("params", obj_conf)

        self.nn = obj_class(**obj_params)

        obj_conf = seotbx.utils.get_key("loss", model_config)
        obj_type = seotbx.utils.get_key("type", obj_conf)
        obj_class = seotbx.utils.import_class(obj_type)
        obj_params = seotbx.utils.get_key("params", obj_conf)

        self.loss = obj_class(**obj_params)

        obj_conf = seotbx.utils.get_key("optimizer", model_config)
        obj_type = seotbx.utils.get_key("type", obj_conf)
        obj_class = seotbx.utils.import_class(obj_type)

        obj_params = seotbx.utils.get_key("params", obj_conf)
        obj_params["params"] = self.nn.parameters()
        self.optimizer = obj_class(**obj_params)
        obj_params["params"]=""

        obj_conf = seotbx.utils.get_key("scheduler", model_config)
        obj_type = seotbx.utils.get_key("type", obj_conf)
        obj_class = seotbx.utils.import_class(obj_type)
        obj_params = seotbx.utils.get_key("params", obj_conf)
        obj_params["optimizer"]= self.optimizer
        self.scheduler = obj_class(**obj_params)
        obj_params["optimizer"] = ""

        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()

        obj_conf = seotbx.utils.get_key("datasets", model_config)
        datasets_keys = seotbx.utils.get_key("datasets_keys", obj_conf)
        self.datasets = {}

        for dataset_key in ["train_dataset", "val_dataset", "test_dataset"]:

            if dataset_key == "test_dataset":
                continue

            dataset_id = datasets_keys[dataset_key]
            ds_config = seotbx.utils.get_key(dataset_id, obj_conf)
            obj_type = seotbx.utils.get_key("type", ds_config)
            obj_params = seotbx.utils.get_key("params", ds_config)

            transforms_configs = seotbx.utils.get_key_def("transform", obj_params, default=[])
            transforms = []
            for transform_conf in transforms_configs:
                tr_obj_type = seotbx.utils.get_key("type", transform_conf)
                tr_obj_params = seotbx.utils.get_key_def("params", transform_conf, default={})
                tr_obj_class = seotbx.utils.import_class(tr_obj_type)
                transforms.append(tr_obj_class(**tr_obj_params))

            obj_params["transform"] = torchvision.transforms.Compose(transforms)

            obj_class = seotbx.utils.import_class(obj_type)
            self.datasets[dataset_key] = obj_class(**obj_params)

        print(self.hparams)



    def forward(self, x):
        return self.nn.forward(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'train_loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        tensorboard_logs = {'train_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        tensorboard_logs = {'test_loss': loss}
        return {'test_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.datasets["train_dataset"], batch_size=1, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.datasets["val_dataset"], batch_size=1, num_workers=4)


from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


class BaseTrainer():

    def __init__(self, config, **kwargs):
        obj_conf = seotbx.utils.get_key("trainer", config)
        super().__init__()

        obj_type = seotbx.utils.get_key("type", obj_conf)
        obj_class = seotbx.utils.import_class(obj_type)
        obj_params = seotbx.utils.get_key("params", obj_conf)

        self.trainer = obj_class(**obj_params)

    def fit(self, model: LightningModule,
            train_dataloader: Optional[DataLoader] = None,
            val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
            datamodule: Optional[LightningDataModule] = None, ):
        return self.trainer.fit(model, train_dataloader, val_dataloaders, datamodule)


def train_application_func(args):
    save_dir = args.save_dir

    with open("/misc/voute1_ptl-bema1/visi/beaulima/projects/OGC/eo-tbx/seotbx/seotbx/configs/polsar.json", 'r') as fp:
        import json
        config = json.load(fp)

    model = BaseModel(config)
    #data = BaseData(config)
    trainer = Trainer(gpus=1)

    trainer.fit(model=model)

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = save_dir

    # each LightningModule defines arguments relevant to it
    parser = Trainer.from_argparse_args(args)
    # parser.set_defaults(gpus=2)
    # args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)
    return


"""
Example template for defining a system.
"""
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class LightningTemplateModel(LightningModule):
    """
    Sample model to show how to define a template.
    Example:
        >>> # define simple Net for MNIST dataset
        >>> params = dict(
        ...     in_features=28 * 28,
        ...     hidden_dim=1000,
        ...     out_features=10,
        ...     drop_prob=0.2,
        ...     learning_rate=0.001 * 8,
        ...     batch_size=2,
        ...     data_root='./datasets',
        ...     num_workers=8,
        ... )
        >>> model = LightningTemplateModel(**params)
    """

    def __init__(self,
                 in_features: int = 28 * 28,
                 hidden_dim: int = 1000,
                 out_features: int = 10,
                 drop_prob: float = 0.2,
                 learning_rate: float = 0.001 * 8,
                 batch_size: int = 2,
                 data_root: str = './datasets',
                 num_workers: int = 8,
                 **kwargs
                 ):
        # init superclass
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()

        self.c_d1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim,

                              out_features=self.hparams.out_features)

        self.example_input_array = torch.zeros(2, 1, 28, 28)

    def forward(self, x):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        x = self.c_d1(x.view(x.size(0), -1))
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        MNIST(self.hparams.data_root, train=True, download=True, transform=transforms.ToTensor())
        MNIST(self.hparams.data_root, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        self.mnist_train = MNIST(self.hparams.data_root, train=True, download=False, transform=transform)
        self.mnist_test = MNIST(self.hparams.data_root, train=False, download=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--hidden_dim', default=50000, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--out_features', default=10, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)
        parser.add_argument('--num_workers', default=4, type=int)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        return parser
