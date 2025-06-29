# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2024/03/27 11:43:56
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : Blended.py
# @Description  : xxxx
# sys
import os
import sys
# mindspore
import mindspore
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.nn import cosine_decay_lr
import mindspore.dataset.vision as vision
from mindspore.dataset import GeneratorDataset

# core
from .Attack import Attack
from ..Base import *
# numpy
import numpy as np
from PIL import Image
import math
import math
import random

def save_dataset(dataset, poison_datasets_path):
    np.savez(
        poison_datasets_path,
        data=dataset.data,
        targets=dataset.targets,
        classes = dataset.classes,
        modified_targets=dataset.modified_targets,
        poison_indices=dataset.poison_indices,
        y_target=dataset.y_target,
        W = dataset.W,
        H = dataset.H,
        poisoned_rate=dataset.poisoned_rate,
        pattern_path = dataset.pattern_path,
        pieces = dataset.pieces,
        mask_rate = dataset.mask_rate,
        alpha = dataset.alpha,
        mask = dataset.mask,
        pattern = dataset.pattern,
    )


def load_dataset(poison_datasets_path):
    data_dict = np.load(poison_datasets_path)

    dataset = PoisonedVisionDataset(
        benign_dataset=[], 
        y_target=int(data_dict['y_target']),
        poisoned_rate=float(data_dict['poisoned_rate']),
        pattern_path = str(data_dict['pattern_path']),
        W = data_dict['W'],
        H = data_dict['H'],
        pieces = data_dict['pieces'],
        mask_rate = data_dict['mask_rate'], 
        alpha = data_dict['alpha']
    )

    dataset.data = data_dict['data']
    dataset.targets = data_dict['targets']
    dataset.classes = data_dict['classes']
    dataset.modified_targets = data_dict['modified_targets']
    dataset.poison_indices = data_dict['poison_indices']

    return dataset


class AddVisionDatasetTrigger():
    def __init__(self, pattern, mask, alpha):
        assert pattern is not None, "pattern is None, its shape must be (C, H, W) or (H, W)"
        assert mask is not None, "mask is None, its shape must be (C, W, H) or (W, H)"
        assert alpha is not None, "alpha is None"

        self.pattern = Tensor(pattern, mindspore.float32)
        self.mask = Tensor(mask, mindspore.float32)
        if self.pattern.ndim == 3:
            self.mask = self.mask.expand_dims(0)  # 变成 (1, H, W)

        self.alpha = alpha

    def add_trigger(self, img):
        img = img + self.alpha * self.mask * (self.pattern - img)
        return img

    def __call__(self, img):
        img = self.add_trigger(img)
        return img


class PoisonedVisionDataset:
    """
    Poisoned VisionDataset : add the trigger generation logic to the sample.
    Keep interface compatible with the original PyTorch version.

    Args:
        pattern (None | Tensor): shape (1, W, H) or (W, H).
        pieces (int)
        mask_rate (float)
        alpha (float)
    """

    def __init__(self, 
                benign_dataset,
                y_target,
                poisoned_rate,
                pattern_path, 
                W, H, 
                pieces, 
                mask_rate, 
                alpha):
        
        super(PoisonedVisionDataset, self).__init__()

        def convert_to_numpy(dataset):
            data_list = []
            label_list = []
            for data in dataset.create_dict_iterator():
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

        self.pattern_path = pattern_path
        self.pieces = pieces
        self.mask_rate = mask_rate
        self.alpha = alpha

        self.W = W
        self.H = H

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # print(f"BASE_DIR:{BASE_DIR}\n")

        # print(f"BASE_DIR:{type(BASE_DIR)}, pattern_path:{type(pattern_path)}\n")

        pattern_path = os.path.join(BASE_DIR,pattern_path)
        # print(f"pattern_path:{pattern_path}\n")

        trigger = Image.open(pattern_path).convert("RGB")
        resized_trigger = trigger.resize((W, H))
        pattern_np = np.array(resized_trigger).transpose((2, 0, 1)) / 255.0  # (C, H, W)

        self.pattern = pattern_np.astype(np.float32)

        self.mask = self.get_trigger_mask(resized_trigger.size[0], self.pieces, int(self.pieces * self.mask_rate))
        self.poison_indices = None
        self.modified_targets = None
        self.set_poisoned_subdatasets(y_target=self.y_target, poisoned_rate=self.poisoned_rate)

    def set_poisoned_subdatasets(self, y_target=None, poisoned_rate=None):
        total_num = len(self.data)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should >= 0'

        tmp_list = np.arange(len(self.data))[~np.array(self.targets == y_target)]
        random.shuffle(tmp_list)

        self.poison_indices = sorted(list(tmp_list[:poisoned_num]))
        self.modified_targets = np.array(deepcopy(self.targets))
        self.modified_targets[self.poison_indices] = y_target


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target, index = self.get_sample_by_index(index)
        img = img.squeeze()
        if len(img.shape) == 3 and img.shape[0] == 32:
            img = np.transpose(img, (2, 0, 1)) # 输出形状 (3, 224, 224)
            
        # print(f"index:{index}, img:{img},img.shape:{img.shape}\n")  

        return img, target, index

    def get_sample_by_index(self, index):
        """
        Add watermarked trigger to sample if index is poisoned.
        Return (img, target, index)
        """
        assert self.pattern is not None, "pattern is None"
        assert self.mask is not None, "mask is None"
        assert self.alpha is not None, "alpha is None"

        pattern = self.pattern
        mask = self.mask
        alpha = self.alpha

        img = self.data[index].astype(np.float32) / 255.0
        target = int(self.modified_targets[index])

        if img.shape[0] == 32: 
            img = img.transpose(2, 0, 1) 

        if index in self.poison_indices:
            img = img + alpha * mask * (pattern - img)
            img = np.clip(img, 0.0, 1.0)
            target = self.y_target

        return img, target, index
    

    def get_trigger_mask(self, img_size, total_pieces, masked_pieces):
        """
        Return mask (Tensor), shape (H, W), values 0 or 1.
        """

        div_num = int(math.sqrt(total_pieces))
        step = int(img_size // div_num)
        candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
        mask_np = np.zeros((img_size, img_size), dtype=np.float32)

        for i in candidate_idx:
            x = int(i // div_num)
            y = int(i % div_num)
            mask_np[x * step: (x + 1) * step, y * step: (y + 1) * step] = 1.0

        mask_np = np.expand_dims(mask_np, axis=0).astype(np.float32)
    
        return mask_np

    # Accessor methods:
    def get_classes(self):
        return self.classes

    def get_y_target(self):
        return self.y_target

    def get_poisoned_rate(self):
        return self.poisoned_rate

    def modify_targets(self, indices, labels):
        self.modified_targets[indices] = labels

    def get_real_targets(self):
        return self.targets

    def get_modified_targets(self):
        return self.modified_targets

    def get_poison_indices(self):
        return self.poison_indices

    def generator(self):
        """
        Generator used for MindSpore GeneratorDataset.
        Yields (img, target, index)
        """
        for index in range(len(self)):
            img, target, index = self.get_sample_by_index(index)
            yield img.asnumpy(), target, index

    def get_dataset(self):
        """
        Return MindSpore GeneratorDataset object
        """
        return GeneratorDataset(source=self.generator, column_names=["image", "label", "index"], shuffle=False)

def create_train_loader(dataset, batch_size, num_workers):
    def generator():
        for idx in range(len(dataset)):
            yield dataset[idx]  # 这会返回 (img, target, index)

    return GeneratorDataset(
        source=generator,
        column_names=["data", "label", "index"],
        shuffle=True,
        num_parallel_workers=1
    ).batch(batch_size, drop_remainder=False)

class Blended(Base, Attack):
    def __init__(self, task, attack_schedule):
        schedule = None
        if 'train_schedule' in attack_schedule:
            schedule = attack_schedule['train_schedule']
        Base.__init__(self, task, schedule=schedule)
        Attack.__init__(self)
        self.attack_schedule = attack_schedule
        assert 'attack_strategy' in self.attack_schedule, "Attack_config must contain 'attack_strategy' configuration! "
        self.attack_strategy = attack_schedule['attack_strategy']
        
    def get_attack_strategy(self):
        return self.attack_strategy

    def create_poisoned_dataset(self, dataset, y_target=None, poisoned_rate=None, train=True):
       
        benign_dataset = dataset
        if y_target is None:
            y_target = self.attack_schedule['y_target']
        if poisoned_rate is None:
            poisoned_rate = self.attack_schedule['poisoned_rate']

        pattern_path = self.attack_schedule['pattern']
        W = self.attack_schedule['W']
        H = self.attack_schedule['H']
        pieces = self.attack_schedule['pieces']
        mask_rate = self.attack_schedule['mask_rate']
        alpha = self.attack_schedule['alpha']

        msg = "\n\n\n==========Start creating poisoned_dataset==========\n"
        print(msg)
        msg = f"Total samples: {len(benign_dataset)},Among the poisoned samples:{int(len(benign_dataset) * poisoned_rate)}\n"
        print(msg)

        poisoned_dataset = PoisonedVisionDataset(
            benign_dataset, y_target, poisoned_rate,
            pattern_path, W, H, pieces, mask_rate, alpha
        )
        return poisoned_dataset
    
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

        train_loader = GeneratorDataset(
            train_dataset,
            column_names=["data", "label","index"],
            shuffle=True,
            num_parallel_workers=current_schedule['num_workers']
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
        
        scheduler = None

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
            lr = self.adjust_learning_rate(optimizer, i, current_schedule)
            for step, (batch_img, batch_label, _) in enumerate(train_loader):
                
                batch_img = ops.cast(batch_img, ms.float32)
                batch_label = ops.cast(batch_label, ms.int32)

                if isinstance(batch_label, Tensor):
                    batch_label = batch_label.astype(ms.int32)
                else:
                    batch_label = Tensor(batch_label, dtype=ms.int32)

                loss = train_step(batch_img, batch_label)
                iteration += 1            

                log_iteration_interval = current_schedule['log_iteration_interval']
                # log_iteration_interval = 10
                  
                global_step = i * steps_per_epoch + step
                current_lr = lr_tensor[global_step]
                if iteration % log_iteration_interval == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+resumed_epoch+1}/{current_schedule['epochs']}, iteration:{step + 1}\{len(train_dataset)//current_schedule['batch_size']}, lr:{current_lr}, loss: {float(loss)}, time: {time.time()-last_time}\n"
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
            model = self.model
        else:
            model = model
        self.model.set_train(False)

        test_loader = create_train_loader(dataset, batch_size, num_workers)

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
  



 
