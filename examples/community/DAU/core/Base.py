# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:26:03
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : Base.py
# @Description  : Logical implementation of model training and testing
# sys
import os
import os.path as osp
import sys
import time
from copy import deepcopy

# mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.train import Model
from mindspore.train.callback import Callback
from mindspore.common.initializer import initializer

# core
from abc import abstractmethod
# numpy
import random
import numpy as np

# utils
from utils import compute_accuracy
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from  utils import Log, log
from tqdm import tqdm

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# support_list = (
#     'DatasetFolder',
#     'MNIST',
#     'CIFAR10',
#     'VisionDataset'
# )

class Subset:

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def get_resumed_epoch(text):
    import re
    match = re.search(r"epoch_(\d+)", text)
    if match:
        epoch_number = match.group(1)
        print("Epoch number:", epoch_number)
        return int(epoch_number)

class Base():
    def __init__(self, task=None, schedule=None):
        if task is not None:
            self.train_dataset = task['train_dataset']
            self.test_dataset = task['test_dataset']
            assert 'model' in task, "task must contain 'model' configuration! "
            self.model = task['model'] 
            self.init_params = deepcopy(self.model.parameters_dict())
            assert 'loss' in task, "task must contain 'loss' configuration! "
            self.loss = task['loss']
            assert 'optimizer' in task, "task must contain 'optimizer' configuration! "
            self.optimizer = task['optimizer']
            self.lr_scheduler = None
            if "lr_scheduler" in task.keys():
                self.lr_scheduler = task['lr_scheduler']
        
        self.global_schedule = deepcopy(schedule)  
        self.current_schedule = None 
        if schedule is not None:
            assert 'seed' in schedule, "task must contain 'seed' configuration! "
            assert 'deterministic' in schedule, "task must contain 'deterministic' configuration! "
            if 'seed' in schedule and schedule['seed'] is not None and 'deterministic' in schedule and schedule['deterministic']: 
                self._set_seed(schedule['seed'], schedule['deterministic'])

    def _set_seed(self, seed, deterministic):
        ms.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            context.set_context(deterministic='ON')

    def init_model(self):
        for name, param in self.model.parameters_and_names():
            if name in self.init_params:
                param.set_data(self.init_params[name])

    def get_model(self):
        return deepcopy(self.model)

    def set_train_dataset(self, dataset):
        self.train_dataset = dataset
        
    def set_test_dataset(self, dataset):
        self.test_dataset = dataset

    def get_dataset(self):
        return self.train_dataset, self.test_dataset

    def adjust_learning_rate(self, optimizer, epoch, current_schedule):
        if "schedule" in current_schedule.keys() and current_schedule["schedule"] is not None:
            if epoch in current_schedule["schedule"]:
                new_lr = current_schedule['lr'] * current_schedule["gamma"]
                current_schedule['lr'] = new_lr 
                ops.assign(optimizer.learning_rate, current_schedule['lr'])
                log("Adjust learning rate to {}\n".format(new_lr))
        return current_schedule['lr']

    def train(self, train_dataset=None, test_dataset=None, schedule=None):
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_schedule is not None:
            current_schedule = deepcopy(self.global_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
            
        if train_dataset is None:
            train_dataset = self.train_dataset

        resumed_epoch = 0
        if 'pre_train_model' in current_schedule and current_schedule['pre_train_model'] is not None:
            print(current_schedule['pre_train_model'])
            resumed_epoch = get_resumed_epoch(current_schedule['pre_train_model'])
            ms.load_param_into_net(self.model, ms.load_checkpoint(current_schedule['pre_train_model']))

        # Set device
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

        print(f"Use CPU to train.\n")

        train_loader = GeneratorDataset(
            train_dataset,
            column_names=["data", "label"],
            shuffle=True,
            num_parallel_workers=current_schedule['num_workers']
        ).batch(current_schedule['batch_size'], drop_remainder=False)

        optimizer = self.optimizer(self.model.trainable_params(), 
                                 learning_rate=current_schedule['lr'], 
                                 momentum=current_schedule['momentum'], 
                                 weight_decay=current_schedule['weight_decay'])
        
        scheduler = None
        # if self.lr_scheduler is not None and current_schedule['lr_scheduler_config'] is not None:
        #     scheduler = self.lr_scheduler(optimizer, **current_schedule['lr_scheduler_config'])

        experiment = current_schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute model train in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)

        iteration = 0
        last_time = time.time()
        
        msg = f"Total train samples: {len(train_dataset)}\nBatch size: {current_schedule['batch_size']}\niteration every epoch: {len(train_dataset) // current_schedule['batch_size']}\nInitial learning rate: {current_schedule['lr']}\n"
        log(msg)

        start_time = time.time()
        
        # Define train step function
        def forward_fn(data, label):
            logits = self.model(data)
            # print("logits shape:", logits.shape, "dtype:", logits.dtype)  # 应为 (batch, num_classes)
            # print("labels shape:", label.shape, "dtype:", label.dtype)  # 应为 (batch,) 且 dtype=int32

            loss = self.loss(logits, label)
            return loss, logits
        
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            return loss
        
        for i in range(current_schedule['epochs'] - resumed_epoch):
            lr = self.adjust_learning_rate(optimizer, i, current_schedule)
            for batch_id, (batch_img, batch_label) in enumerate(train_loader):
                
                batch_label = ops.cast(batch_label, ms.int32)
                loss = train_step(batch_img, batch_label)
                iteration += 1

                if iteration % current_schedule['log_iteration_interval'] == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+resumed_epoch+1}/{current_schedule['epochs']}, iteration:{batch_id + 1}\{len(train_dataset)//current_schedule['batch_size']},lr: {lr}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    log(msg)

            if scheduler is not None:
                scheduler.step()
                log("Adjust learning rate to {}\n".format(optimizer.learning_rate.asnumpy()))

            # test_iteration_interval = current_schedule["test_iteration_interval"]
            test_iteration_interval = 1
            if test_dataset is not None and (i + resumed_epoch + 1) % test_iteration_interval == 0:

                predict_digits, labels = self._test(test_dataset, model=deepcopy(self.model))
                # predict_digits = ms.Tensor(predict_digits, dtype=ms.float32)
                # labels = ms.Tensor(labels, dtype=ms.int32)
                # print(f"type(predict_digits):{type(predict_digits)},predict_digits:{predict_digits.shape}\n")
                # print(f"type(labels):{type(labels)},len(labels):{len(labels)},labels:{labels}\n")

                acc = compute_accuracy(predict_digits, labels, topk=(1,3,5))
                log("Total samples:{0},  benign_acc:{1}\n".format(len(test_dataset), acc[0]))                                                                                                                                                          

                poisoned_test_indexs = test_dataset.get_poison_indices()
                benign_test_indexs = list(set(range(len(test_dataset))) - set(poisoned_test_indexs))
                poisoned_test_indexs = list(poisoned_test_indexs)
                
                predict_digits, labels = self._test(test_dataset, model=deepcopy(self.model))
                benign_acc = compute_accuracy(predict_digits[benign_test_indexs], labels[benign_test_indexs], topk=(1,3,5))
                poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs], labels[poisoned_test_indexs], topk=(1,3,5))
    
                log("Benign_accuracy:{0}, poisoning_accuracy:{1}\n".format(benign_acc, poisoned_acc))

                # log("test data with target label\n")
                # targets = test_dataset.get_modified_targets()
                # y_target = test_dataset.get_y_target()
                # y0_indices = np.where(targets==y_target)[0]
                # clean_y0_indices = np.intersect1d(y0_indices, benign_test_indexs)
                # poison_y0_indexs = list(set(y0_indices)- set(clean_y0_indices))
                
                # predict_digits, labels = self._test(Subset, model=deepcopy(self.model))
                # benign_acc = compute_accuracy(predict_digits, labels, topk=(1,3,5))
                
                # predict_digits, labels = self._test(Subset(test_dataset, poison_y0_indexs), model=deepcopy(self.model))
                # poisoned_acc = compute_accuracy(predict_digits, labels, topk=(1,3,5))
                
                # log("Total samples:{0}, poisoning samples:{1}, benign samples:{2}\n".format(len(y0_indices), len(poison_y0_indexs), len(clean_y0_indices)))                                                                                                                                                
                

        end_time = time.time()
        log(f"Train end, spent time(s):{end_time - start_time}\n")

    def _test(self, dataset, model=None, batch_size=16, num_workers=8):
        if model is None:
            model = self.model
        else:
            model = model

        test_loader = GeneratorDataset(
            dataset,
            column_names=["data", "label"],
            shuffle=False,
            num_parallel_workers=num_workers
        ).batch(batch_size, drop_remainder=False)

        predict_digits = []
        labels = []
        
        for batch in tqdm(test_loader, desc='testing', unit='batch'):
            batch_img, batch_label = batch[0], batch[1]
            batch_pred = model(batch_img)
            predict_digits.append(batch_pred.asnumpy())
            labels.append(batch_label.asnumpy())
        
        predict_digits = np.concatenate(predict_digits, axis=0)
        labels = np.concatenate(labels, axis=0)                 

        predict_digits = ms.Tensor(predict_digits, dtype=ms.float32)
        labels = ms.Tensor(labels, dtype=ms.int32)
        return predict_digits, labels
    
    def test(self, model=None, test_dataset=None, schedule=None):
        if schedule is not None:
            current_schedule = schedule
        elif self.global_schedule is not None:
            current_schedule = self.global_schedule
        else:
            raise AttributeError("Test schedule is None, please check your schedule setting.")

        if model is None:
            model = self.model

        if test_dataset is None:
            test_dataset = self.test_dataset

        # Set device
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        print(f"Use CPU to test.")

        experiment = current_schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute model test in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)
        
        last_time = time.time()
        predict_digits, labels = self._test(test_dataset, model=deepcopy(model), 
                                          batch_size=current_schedule['batch_size'], 
                                          num_workers=current_schedule['num_workers'])

        total_num = labels.shape[0]


        prec1, prec5 = compute_accuracy(predict_digits, labels, topk=(1, 5))
        top1_correct = int(round(prec1.item() / 100.0 * total_num))
        top5_correct = int(round(prec5.item() / 100.0 * total_num)) 
        
        msg = "\n==========Test result on test dataset==========\n"
        log(msg)
        msg = f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
        log(msg)
        
        return predict_digits, labels