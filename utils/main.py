import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn

from dataset import dataset_loader, augmentation
from utils import forward, tools
from models import create_model
import loss

from tqdm import tqdm
import numpy as np
import random
import time
import os


class main:
    def __init__(self, args, seed=1):
        # Read parser.
        self.args = args
        self.seed = seed
        self.lr = self.args.batch_size * self.args.lr
        self.batch_size = self.args.batch_size * self.args.gpu_num
        self.print_info()

    def print_info(self):
        print('Train: {}, Data: {}, Size: {}'.format(
               self.args.train, self.args.task, self.args.size))
        print('Epochs: {}, Batch size: {}, Batch size per GPU: {}, LR: {}'.format(
               self.args.epochs, self.batch_size, self.args.batch_size, self.lr))
        print('Model: {}, Method: {}'.format(self.args.model_name, self.args.method))
        print('Hyper-parameters: tau: {}, pow: {}, adj_tpye: {}'.format(self.args.tau, 
                                                                        self.args.pow, 
                                                                        self.args.adj_type))

    def check_path(self):
        self.save_path = self.args.save_dir
        if os.path.exists(self.save_path) is not True:
            os.mkdir(self.save_path)

        rows = [
                self.args.task, 
                str(self.args.size),
                self.args.method, 
                self.args.model_name
                ]

        for row in rows:
            self.save_path = os.path.join(self.save_path, row)
            if os.path.exists(self.save_path) is not True:
                os.mkdir(self.save_path)

        name = 'sd{}_ep{}_bs{}'.format(self.seed, self.args.epochs, self.args.batch_size)
        # name = 'sd{}_bs{}'.format(self.seed, self.args.batch_size)
        if self.args.tau:
            name = name + '_t{}'.format(self.args.tau)
        if self.args.pow:
            name = name + '_p{}_adj{}'.format(self.args.pow, self.args.adj_type)
        self.save_path = os.path.join(self.save_path, name)
        print(f'Checkpoint: {self.save_path}')

    def load_train_data(self):
        train_transforms = augmentation(resize=self.args.resize, size=self.args.size, is_train=True)
        self.trainset = dataset_loader(
                                self.args.root,
                                task = self.args.task,
                                is_train=True,
                                transforms=train_transforms,
                                in_memory=bool(self.args.in_memory), 
                                )
        self.trainloader = DataLoader(self.trainset, 
                                      batch_size=self.batch_size,
                                      shuffle=True, 
                                      num_workers=self.args.gpu_num*4, 
                                      drop_last=self.args.train,
                                      pin_memory=False)
        assert self.n_class == self.trainset.n_class

        self.label_freq = self.trainset.count
        if self.args.pow:
            self.weight = getattr(tools, 'adjust_type{}'.format(self.args.adj_type))(self.label_freq, self.args.pow)
        else:
            self.weight = None

    def load_valid_data(self):
        valid_transforms = augmentation(resize=self.args.resize, size=self.args.size, is_train=False)
        self.validset = dataset_loader(
                                self.args.root, 
                                task = self.args.task,
                                is_train=False, 
                                transforms=valid_transforms,
                                in_memory=bool(self.args.in_memory), 
                                )
        self.validloader = DataLoader(self.validset, 
                                      batch_size=self.batch_size,
                                      shuffle=False, 
                                      num_workers=self.args.gpu_num*4, 
                                      drop_last=self.args.train,
                                      pin_memory=False)
        self.n_class = self.validset.n_class

    def create_model(self):
        self.model = create_model(backbone=self.args.model_name, 
                                  n_class=self.n_class,
                                  pretrained=self.args.train)
        if self.args.train == 0:
            self.load_check()
        if self.args.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.args.gpu_num)))
        self.model.to(self.args.device)
        cudnn.benchmark = True

    def trainInit(self):
        self.total_loss = getattr(loss, '{}Loss'.format(self.args.method))(tau=self.args.tau, 
                                                                           weight=self.weight)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr=self.lr, momentum=0.9, 
                                         weight_decay=self.args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=0.0)

    def save(self, epoch, train_loss, ce_loss, train_acc, val_acc):
        if self.args.gpu_num > 1:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save({"args": self.args, 
                    "best_epcoh": epoch,
                    "state_dict": model_state_dict,
                    "train_loss": train_loss, 
                    "ce_loss": ce_loss, 
                    "train_acc": train_acc,
                    "val_acc": val_acc 
                   }, self.save_path)

    def load_check(self):
        try:
            model_state_dict = torch.load(self.save_path)
            self.model.load_state_dict(model_state_dict["state_dict"])
            print("Load checkpoints: {}".format(self.save_path))
        except:
            print("No pre-trained checkpoints.")

    def fit(self):
        val_acc = []
        train_loss, ce_loss, train_acc = [], [], []

        acc = forward.validate(self.model, 
                               self.validloader, 
                               device=self.args.device,
                               )
        val_acc.append(acc)

        best_epoch = 0
        # Train the model iteratively.
        for epoch in range(1, self.args.epochs+1):
            loss, ce, acc = forward.trainIter(epoch, 
                                          self.args.epochs, 
                                          self.model, 
                                          self.trainloader, 
                                          self.total_loss, 
                                          self.optimizer, 
                                          device=self.args.device,
                                          )
            train_loss.append(loss)
            ce_loss.append(ce)
            train_acc.append(acc)
            self.scheduler.step()

            if epoch % self.args.eval_epoch == 0:
                acc = forward.validate(self.model, 
                                       self.validloader, 
                                       device=self.args.device,
                                       )
                val_acc.append(acc)
                # Save the best performacne only.
                if acc > max(val_acc[:-1]):
                    best_epoch = epoch
                    print("Best epoch: {}, Validation improve. save as {}"
                          .format(best_epoch, self.save_path))
                    self.save(epoch, train_loss, ce_loss, train_acc, val_acc)
                else:
                    print("Best epoch: {}, Best validation acc: {:.2f}%, checkpoint: {}"
                          .format(best_epoch, 100.0 * max(val_acc[:-1]), self.save_path))
        return max(val_acc)

    def build(self):
        self.load_valid_data()
        self.load_train_data()
        self.check_path()
        self.create_model()

        if self.args.train:
            self.trainInit()
            acc = self.fit()
            return acc
        else:
            try:
                acc = forward.validate_with_fcr(
                                       self.model, 
                                       self.validloader, 
                                       self.label_freq,
                                       device=self.args.device,
                                       )
            except:
                acc = forward.validate(
                                       self.model, 
                                       self.validloader, 
                                       device=self.args.device,
                                       )
            return acc
