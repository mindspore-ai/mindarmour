import mindspore
from mindspore.nn import cosine_decay_lr
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.vision import py_transforms as T
from mindspore.dataset.transforms import py_transforms as C
from .Attack import Attack
from ..Base import *
import numpy as np
from PIL import Image
import random
from copy import deepcopy

def save_dataset(dataset, poison_datasets_path):
    np.savez(
        poison_datasets_path,
        data=dataset.data,
        targets=dataset.targets,
        classes = dataset.classes,
        num_classes = dataset.num_classes,
        train =  dataset.train,
        attack_type =  dataset.attack_type,
        y_target = dataset.y_target,
        poisoned_rate = dataset.poisoned_rate,
        delta = dataset.delta,
        frequency = dataset.frequency,
        poison_indices = dataset.poison_indices,
        modified_targets = dataset.modified_targets
    )

def load_dataset(poison_datasets_path):

    data_dict = np.load(poison_datasets_path)
    dataset = SIGDataset(
        benign_dataset=[], 
        y_target=int(data_dict['y_target']),
        poisoned_rate=float(data_dict['poisoned_rate']),
        delta = data_dict["delta"],
        frequency = data_dict["frequency"], 
        train=data_dict["train"], 
        attack_type=data_dict["attack_type"]
    )
    dataset.data = data_dict['data']
    dataset.targets = data_dict['targets']
    dataset.classes = data_dict['classes']
    dataset.num_classes = data_dict['num_classes']
    dataset.modified_targets = data_dict['modified_targets']
    dataset.poison_indices = data_dict['poison_indices']

    return dataset

def sig(img, delta, freq):
    overlay = np.zeros_like(img, dtype=np.float64)
    _, m, _ = overlay.shape
    for i in range(m):
        overlay[:, i] = delta * np.sin(2 * np.pi * i * freq / m)
    overlay = np.clip(overlay + img, 0, 1.0).astype(np.float32)
    return overlay

class SIGDataset:
    def __init__(self, 
                benign_dataset, 
                y_target,
                poisoned_rate,
                delta, 
                frequency,
                train=True, 
                attack_type='all-to-one'):
        
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

        self.num_classes = len(self.classes)

        self.train = train
        self.attack_type = attack_type

        self.y_target = y_target
        self.poisoned_rate = poisoned_rate
        
        self.delta = delta
        self.frequency = frequency

        self.poison_indices = None
        self.modified_targets = None
        self.set_poisoned_subdatasets(y_target, poisoned_rate)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.modified_targets[index]

        # add trigger
        if index in self.poison_indices:
            img = sig(img, self.delta, self.frequency)


        return img, target, index

    def __len__(self):
        return len(self.data)

    def set_poisoned_subdatasets(self, y_target=None, poisoned_rate=0.0):
        
        self.modified_targets = np.array(deepcopy(self.targets))
        if self.attack_type == "all-to-one":
            if self.train:
                tmp_list = np.where(self.targets == y_target)[0]
            else:
                tmp_list = np.where(self.targets != y_target)[0]
               
        elif self.attack_type == "all-to-all":
            tmp_list = np.arange(len(self.data))

        poisoned_num = int(len(tmp_list) * poisoned_rate)
        assert poisoned_num >= 0

        random.shuffle(tmp_list)
        self.poison_indices = sorted(list(tmp_list[:poisoned_num]))

        if self.attack_type == "all-to-one":
            self.modified_targets[self.poison_indices] = y_target
        elif self.attack_type == "all-to-all":
            self.modified_targets[self.poison_indices] = (self.targets[self.poison_indices] + 1) % self.num_classes

    
    def modify_targets(self, indces, labels):
        self.modified_targets[indces] = labels

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


class SIG(Base, Attack):
    def __init__(self, task, attack_schedule):
        schedule = attack_schedule.get('train_schedule', None)
        Base.__init__(self, task, schedule=schedule)
        Attack.__init__(self)
        self.attack_schedule = attack_schedule
        assert 'attack_strategy' in self.attack_schedule
        self.attack_strategy = self.attack_schedule['attack_strategy']

    def get_attack_strategy(self):
        return self.attack_strategy

    def create_poisoned_dataset(self, dataset, y_target=None, poisoned_rate=None, train=True):
       
        if y_target is None:
            y_target = self.attack_schedule['y_target']
        if poisoned_rate is None:
            poisoned_rate = self.attack_schedule['poisoned_rate']

        delta = self.attack_schedule['delta']
        frequency = self.attack_schedule['frequency']

        return SIGDataset(dataset, y_target, poisoned_rate, delta, frequency, train=train)

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
            column_names=["data", "label", "index"],
            shuffle=True, 
            num_parallel_workers=1
        ).batch(current_schedule['batch_size'], drop_remainder=True)

        steps_per_epoch = len(train_dataset) // current_schedule['batch_size']
        total_steps = current_schedule['epochs'] * steps_per_epoch

        lr_tensor = cosine_decay_lr(
            min_lr=0.001,
            max_lr=0.05,
            total_step=total_steps,
            step_per_epoch=steps_per_epoch,
            decay_epoch=20
        )

        optimizer = self.optimizer(
            self.model.trainable_params(), 
            learning_rate=lr_tensor, 
            momentum=current_schedule['momentum'], 
            weight_decay=current_schedule['weight_decay']
        )

        # optimizer = self.optimizer(self.model.trainable_params(), 
        #     learning_rate=current_schedule['lr'], 
        #     momentum=current_schedule['momentum'], 
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
            # print("logits shape:", logits.shape, "dtype:", logits.dtype)  # 应为 (batch, num_classes)
            # print("labels shape:", label.shape, "dtype:", label.dtype)  # 应为 (batch,) 且 dtype=int32

            loss = self.loss(logits, label)
            return loss, logits
        
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            return loss
        
        # print(f"epochs:{current_schedule['epochs']},resumed_epoch:{resumed_epoch}\n")
        iteration = 0
        self.model.set_train(True)
        for i in range(current_schedule['epochs'] - resumed_epoch):
            lr = self.adjust_learning_rate(optimizer, i, current_schedule)
            for step, (batch_img, batch_label, _) in enumerate(train_loader):
                
                batch_img = ops.cast(batch_img, ms.float32)
                batch_label = ops.cast(batch_label, ms.int32)

                # print(f"batch_img:{batch_img.shape}, len(batch_label):{len(batch_label)}\n")

                loss = train_step(batch_img, batch_label)
                iteration += 1

                log_iteration_interval = current_schedule['log_iteration_interval']
                # log_iteration_interval = 10
                global_step = i * steps_per_epoch + step
                current_lr = lr_tensor[global_step]
                if iteration % log_iteration_interval == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+resumed_epoch+1}/{current_schedule['epochs']}, iteration:{step+1}\{len(train_dataset)//current_schedule['batch_size']}, lr: {current_lr}, loss: {float(loss)}, time: {time.time()-last_time}\n"
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
                
                print(f"len(test_dataset):{len(predict_digits)},benign_test_indexs:{len(benign_test_indexs)},poisoned_test_indexs:{len(poisoned_test_indexs)}\n")

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
