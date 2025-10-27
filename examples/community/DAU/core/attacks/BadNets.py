# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:24:42
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description : This is the implement of BadNets [1] in MindSpore.
#               Reference:[1] Badnets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access 2019.
import copy
import random
import numpy as np
import PIL
from PIL import Image
import mindspore as ms
from mindspore import nn, ops
from mindspore.nn import cosine_decay_lr
# from mindspore.dataset import DatasetNode
from mindspore.dataset.vision import transforms as vision_transforms
from mindspore.dataset.vision import Inter
from mindspore.dataset import Dataset, vision
from .Attack import Attack
from ..Base import *
import core

import decimal
from decimal import Decimal, getcontext
import math

def save_dataset(dataset, poison_datasets_path):
    np.savez(
        poison_datasets_path,
        data=dataset.data,
        targets=dataset.targets,
        classes = dataset.classes,
        modified_targets=dataset.modified_targets,
        poison_indices=dataset.poison_indices,
        y_target=dataset.y_target,
        poisoned_rate=dataset.poisoned_rate,
        pattern = dataset.pattern,
        weight = dataset.weight
    )

def load_dataset(poison_datasets_path):
    data_dict = np.load(poison_datasets_path)
   
    dataset = PoisonedVisionDataset(
        benign_dataset=[], 
        y_target=int(data_dict['y_target']),
        poisoned_rate=float(data_dict['poisoned_rate']),
        pattern=data_dict["pattern"], 
        weight=data_dict["weight"]
    )
    dataset.data = data_dict['data']
    dataset.targets = data_dict['targets']
    dataset.classes = data_dict['classes']
    dataset.modified_targets = data_dict['modified_targets']
    dataset.poison_indices = data_dict['poison_indices']

    return dataset

# support_list = (
#     Dataset,
# )

class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (ms.Tensor): shape (C, H, W).

        Returns:
            ms.Tensor: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res).astype(ms.uint8)

class AddVisionDatasetTrigger():
    def __init__(self, pattern, weight):
        assert pattern is not None, "pattern is None, its shape must be (1, H, w) or (H, W)"
        assert weight is not None, "weight is None, its shape must be  (1, W, H) or(W, H)"

        # self.pattern = pattern
        # if isinstance(self.pattern, np.ndarray):
        #     self.pattern = ms.Tensor(self.pattern)

        # if self.pattern.ndim == 2:
        #     self.pattern = self.pattern.expand_dims(2)

        # self.weight = weight
        # if isinstance(self.weight, np.ndarray):
        #     self.weight= ms.Tensor(self.weight)

        # if self.weight.ndim == 2:
        #     self.weight = self.weight.expand_dims(2)

        
        # # Accelerated calculation
        # self.res = self.weight * self.pattern
        # self.weight = 1.0 - self.weight

        # 转换为 numpy
        self.pattern = pattern.astype(np.float32) if isinstance(pattern, np.ndarray) else np.array(pattern, dtype=np.float32)
        self.weight = weight.astype(np.float32) if isinstance(weight, np.ndarray) else np.array(weight, dtype=np.float32)

        # 确保通道数为 1
        if self.pattern.ndim == 2:
            self.pattern = np.expand_dims(self.pattern, axis=2)  # (1, H, W)
        if self.weight.ndim == 2:
            self.weight = np.expand_dims(self.weight, axis=2)    # (1, H, W)

        # 提前计算好 pattern 部分
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (ms.Tensor): shape (C, H, W).

        Returns:
            ms.Tensor: Poisoned image, shape (C, H, W).
        """

        # print(f"self.weight:{self.weight.shape},img:{img.shape},self.res:{self.res.shape}\n")
        # img = (self.weight * img + self.res).astype(ms.float32)
        
        img = img.astype(np.float32)

        # print(f"self.weight:{self.weight.shape}, self.res:{self.res.shape}, img:{img.shape}\n")

        img = self.weight * img + self.res

        # print(f"img:{img}\n")

        return img

    def __call__(self, img):

        # img_tensor = ms.Tensor(img)
        # img = self.add_trigger(img_tensor)
        # img = img.squeeze()
        # return img

        if not isinstance(img, np.ndarray):
            img = np.array(img, dtype=np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)  # Convert to (C, H, W)
        img = self.add_trigger(img)
        return img

    
class AddMNISTTrigger(AddTrigger):
    def __init__(self, pattern, weight):
        super(AddMNISTTrigger, self).__init__()

        if pattern is None:
            self.pattern = ops.Zeros()((1, 28, 28), ms.uint8)
            self.pattern[0, -2, -2] = 255
        else:
            self.pattern = pattern
            if self.pattern.ndim == 2:
                self.pattern = self.pattern.expand_dims(0)

        if weight is None:
            self.weight = ops.Zeros()((1, 28, 28), ms.float32)
            self.weight[0, -2, -2] = 1.0
        else:
            self.weight = weight
            if self.weight.ndim == 2:
                self.weight = self.weight.expand_dims(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = vision.ToTensor()(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.asnumpy(), mode='L')
        return img

class AddCIFAR10Trigger(AddTrigger):
    def __init__(self, pattern, weight):
        super(AddCIFAR10Trigger, self).__init__()
        if pattern is None:
            self.pattern = ops.Zeros()((1, 32, 32), ms.uint8)
            self.pattern[0, -3:, -3:] = 255
        else:
            self.pattern = pattern
            if self.pattern.ndim == 2:
                self.pattern = self.pattern.expand_dims(0)

        if weight is None:
            self.weight = ops.Zeros()((1, 32, 32), ms.float32)
            self.weight[0, -3:, -3:] = 1.0
        else:
            self.weight = weight
            if self.weight.ndim == 2:
                self.weight = self.weight.expand_dims(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = vision.ToTensor()(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.transpose(1, 2, 0).asnumpy())
        return img

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

class PoisonedVisionDataset:
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight):
        super(PoisonedVisionDataset, self).__init__()
        
        def convert_to_numpy(dataset):
            # print(f"dataset:{type(dataset)}\n")
            data_list = []
            label_list = []
            num = 0
            
            for data in dataset.create_dict_iterator():
                image = data['image'].asnumpy()
                data_list.append(data['image'].asnumpy())
                label_list.append(data['label'].asnumpy())
          
            return np.array(data_list), np.array(label_list)

        if len(benign_dataset) > 0:
            self.data, self.targets = convert_to_numpy(benign_dataset)
        else:
            self.data = []
            self.targets = []

        self.dataset = benign_dataset
        self.classes = np.array(list(set(self.targets)))

        self.y_target = y_target
        self.poisoned_rate = poisoned_rate

        self.pattern = pattern

        self.weight = weight
        self.poison_indices = None
        self.modified_targets = None
        self.set_poisoned_subdatasets(y_target=self.y_target, poisoned_rate=self.poisoned_rate)

        self.AddVisionDatasetTrigger = AddVisionDatasetTrigger(pattern, weight)
        self.transform = benign_dataset.transform if hasattr(benign_dataset, 'transform') else None
        self.target_transform = benign_dataset.target_transform if hasattr(benign_dataset, 'target_transform') else None

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        img, target, index = self.get_sample_by_index(index)

        img = img.squeeze()
        # if img.dim() == 3 and img.shape[0] == 32:
        if len(img.shape) == 3 and img.shape[0] == 32:
            img = np.transpose(img, (2, 0, 1)) # 输出形状 (3, 224, 224)
       
        # print(f"index:{index},img:{type(img)}\n")

        return img, target, index
       
    def get_sample_by_index(self, index):
        img, target = self.data[index], int(self.modified_targets[index])
        
        # print(f"index:{index},img.shape:{img.shape}\n")

        if len(img.shape) == 2:
            img = img.squeeze()
        elif len(img.shape) == 3 and img.shape[0] == 3:
           img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])

        if index in self.poison_indices:
            img = self.AddVisionDatasetTrigger(img)
            
        # else:
        #     img = ms.Tensor(img)
         
        return img, target, index
    

    def set_poisoned_subdatasets(self, y_target=None, poisoned_rate=0.0):
        poisoned_num = int(len(self.data) * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        
        tmp_list = np.arange(len(self.data))[~np.array(self.targets == y_target)]
        random.shuffle(tmp_list)
        self.poison_indices = sorted(list(tmp_list[:poisoned_num]))
        self.modified_targets = np.array(deepcopy(self.targets))
        self.modified_targets[self.poison_indices] = y_target
   
    def modify_targets(self, indices, labels):
        self.modified_targets[indices] = labels
        
    def get_real_targets(self):
        return self.targets
    
    def get_classes(self):
        return self.classes
    
    def get_y_target(self):
        return self.y_target
    
    def get_poisoned_rate(self):
        return self.poisoned_rate
    
    def get_poison_indices(self):
        return self.poison_indices
    
    def get_modified_targets(self):
        return self.modified_targets
    
# def create_train_loader(dataset, batch_size, num_workers):
#     def generator():
#         for idx in range(len(dataset)):
#             yield dataset[idx]  # 这会返回 (img, target, index)

#     return GeneratorDataset(
#         source=generator,
#         column_names=["data", "label", "index"],
#         shuffle=True,
#         num_parallel_workers=1
#     ).batch(batch_size, drop_remainder=False)

class BadNets(Base, Attack):
    def __init__(self, task, attack_schedule):
        schedule = None
        if 'train_schedule' in attack_schedule:
            schedule = attack_schedule['train_schedule']
        Base.__init__(self, task, schedule=schedule)   
        Attack.__init__(self)
        self.attack_schedule = attack_schedule
        assert 'attack_strategy' in self.attack_schedule, "Attack_config must contain 'attack_strategy' configuration!"
        self.attack_strategy = attack_schedule['attack_strategy']
    
    def get_attack_strategy(self):
        return self.attack_strategy
    
    def create_poisoned_dataset(self, dataset, y_target=None, poisoned_rate=None, train=True):
        benign_dataset = dataset

        if y_target is None:
            assert 'y_target' in self.attack_schedule, "Attack_config must contain 'y_target' configuration!"
            y_target = self.attack_schedule['y_target']

        if poisoned_rate is None:
            assert 'poisoned_rate' in self.attack_schedule, "Attack_config must contain 'poisoned_rate' configuration!"
            poisoned_rate = self.attack_schedule['poisoned_rate']

        assert 'pattern' in self.attack_schedule, "Attack_config must contain 'pattern' configuration!"
        pattern = self.attack_schedule['pattern']
        assert 'weight' in self.attack_schedule, "Attack_config must contain 'weight' configuration!"
        weight = self.attack_schedule['weight']
      
        # dataset_type = type(benign_dataset)
        # assert dataset_type in support_list or isinstance(benign_dataset, Dataset), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'

        print(f"\n======================Create_poisoned_dataset==============\n")
        msg = f"Total samples: {len(benign_dataset)}, Among the poisoned samples: {int(len(benign_dataset) * poisoned_rate)}\n"
        print(msg)

        return PoisonedVisionDataset(benign_dataset, y_target, poisoned_rate, pattern, weight)
    
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
        print(f"Use CPU to train.")

        # train_loader = create_train_loader(
        #     train_dataset, 
        #     current_schedule['batch_size'], 
        #     current_schedule['num_workers']
        # )

        train_loader = GeneratorDataset(
            train_dataset, 
            column_names=["data", "label", "index"],
            shuffle=True, 
            num_parallel_workers=1
        ).batch(current_schedule['batch_size'], drop_remainder=True)

        steps_per_epoch = len(train_dataset) // current_schedule['batch_size']
        total_steps = current_schedule['epochs'] * steps_per_epoch

        lr_tensor = cosine_decay_lr(
            min_lr=0.001,
            max_lr=0.01,
            total_step=total_steps,
            step_per_epoch=steps_per_epoch,
            decay_epoch=50
        )

        optimizer = self.optimizer(
            self.model.trainable_params(), 
            learning_rate=lr_tensor, 
            momentum=current_schedule['momentum'], 
            weight_decay=current_schedule['weight_decay']
        )
        
        # optimizer = self.optimizer(
        #     self.model.trainable_params(), 
        #     learning_rate=lr_tensor, 
        #     # momentum=current_schedule['momentum'], 
        #     weight_decay=current_schedule['weight_decay']
        # )
        
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
            loss = self.loss(logits, label)
            return loss, logits
        
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            return loss
        
        # print(f"epochs:{current_schedule['epochs']},resumed_epoch:{resumed_epoch}\n")

        self.model.set_train(True)
        for i in range(current_schedule['epochs'] - resumed_epoch):
            for step, (batch_img, batch_label, _) in enumerate(train_loader):
                
                # print(f"batch_img:{type(batch_img)}\n")

                batch_img = ops.cast(batch_img, ms.float32)
                batch_label = ops.cast(batch_label, ms.int32)

                loss = train_step(batch_img, batch_label)
                iteration += 1
                 
                global_step = i * steps_per_epoch + step
                current_lr = lr_tensor[global_step]

                log_iteration_interval = current_schedule['log_iteration_interval']
                # log_iteration_interval = 10
                if iteration % log_iteration_interval == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+resumed_epoch+1}/{current_schedule['epochs']}, iteration:{step + 1}\{len(train_dataset)//current_schedule['batch_size']}, lr:{current_lr}, loss:{float(loss)}, time: {time.time()-last_time}\n"
                    log(msg)

            if scheduler is not None:
                scheduler.step()
                log("Adjust learning rate to {}\n".format(optimizer.learning_rate.asnumpy()))

            # test_iteration_interval = current_schedule["test_iteration_interval"]
            test_iteration_interval = 1
            if test_dataset is not None and (i + resumed_epoch + 1) % test_iteration_interval == 0:

                print("==========Test result on poisoned test dataset==========")
                
                predict_digits, labels = self._test(test_dataset, model=deepcopy(self.model))

                poisoned_test_indexs = test_dataset.get_poison_indices()
                benign_test_indexs = list(set(range(len(test_dataset))) - set(poisoned_test_indexs))
                poisoned_test_indexs = poisoned_test_indexs.tolist()
                
                benign_acc = compute_accuracy(predict_digits[benign_test_indexs], labels[benign_test_indexs], topk=(1,3,5))
                poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs], labels[poisoned_test_indexs], topk=(1,3,5))
    
                log("Total samples:{0}, poisoning samples:{1}, benign samples:{2}\n".format(len(test_dataset), len(poisoned_test_indexs), len(benign_test_indexs)))
                log("Benign_accuracy:{0}, poisoning_accuracy:{1}\n".format(benign_acc[0], poisoned_acc[0]))


        end_time = time.time()
        log(f"Train end, spent time(s):{end_time - start_time}\n")

    def _test(self, dataset, model=None, batch_size=128, num_workers=8):
        if model is None:
            model = deepcopy(self.model)
        else:
            model = model

        model.set_train(False)

        test_loader = GeneratorDataset(
            dataset, 
            column_names=["data", "label", "index"],
            shuffle=True, 
            num_parallel_workers=1
        ).batch(batch_size, drop_remainder=False)

        # create_train_loader(dataset, batch_size, num_workers)

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
  