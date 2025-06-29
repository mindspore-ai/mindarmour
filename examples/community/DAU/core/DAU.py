import mindspore 
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context
from mindspore.dataset import GeneratorDataset, WeightedRandomSampler
import core
from .Base import Base
from .Defense import Defense
import numpy as np
import json
import time
from copy import deepcopy
from utils import compute_indexes, compute_confusion_matrix
from  utils import Log, log, compute_accuracy
from tqdm import tqdm

def adjust_learning_rate(optimizer, epoch, current_schedule):
    if "schedule" in current_schedule.keys() and current_schedule["schedule"] is not None:
        if epoch in current_schedule["schedule"]:
            current_schedule['lr'] *= current_schedule['gamma']
            for param in optimizer.parameters:
                param.set_data(param * current_schedule['lr'])
            log("Adjust learning rate to {}\n".format(current_schedule['lr']))
    return current_schedule

def evaluate_filter(dataset, predict_poisoned_indices):
    poison_indices = dataset.get_poison_indices()
    precited = np.zeros(len(dataset))
    precited[predict_poisoned_indices] = 1
    expected = np.zeros(len(dataset))
    expected[poison_indices] = 1
    tp, fp, tn, fn = compute_confusion_matrix(precited, expected)
    return tp, fp, tn, fn

class EntropyLoss(nn.Cell):
    def __init__(self, prob_min=1.0e-7, num_classes=10, reduction="mean"):
        super(EntropyLoss, self).__init__()
        self.prob_min = prob_min
        self.num_classes = num_classes
        self.reduction = reduction
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()

    def construct(self, predict, target):
        prob_matrix = self.softmax(predict)
        prob_matrix = ops.clip_by_value(prob_matrix, self.prob_min, 1.0)
        log_likelihood_matrix = self.log_softmax(predict)

        if self.reduction == "mean":
            loss = -1.0 * ops.mean(ops.sum(log_likelihood_matrix * prob_matrix, 1), 0)
        elif self.reduction == "sum":
            loss = -1.0 * ops.sum(log_likelihood_matrix * prob_matrix)
        return loss

class ADVLoss(nn.Cell):
    def __init__(self, prob_min=1e-7, num_classes=10, reduction="mean"):
        super(ADVLoss, self).__init__()
        self.prob_min = prob_min
        self.num_classes = num_classes
        self.reduction = reduction
       
    def construct(self, x, target):
        prob = ops.softmax(x, axis=-1)
        prob = ops.clip_by_value(prob, self.prob_min, 1.0)
        one_hot = ops.one_hot(target, self.num_classes, Tensor(1.0), Tensor(0.0))
        one_hot = ops.clip_by_value(one_hot, self.prob_min, 1.0)
        sub_prob = ops.clip_by_value(1.0 - prob, self.prob_min, 1.0)
        loss = -1.0 * ops.sum(one_hot * ops.log(sub_prob), -1)
        if self.reduction == "mean":
            loss = ops.mean(loss)
        elif self.reduction == "sum":
            loss = ops.sum(loss)
        return loss

class BCELoss(nn.Cell):
    def __init__(self, prob_min=1e-7, num_classes=10, reduction="mean"):
        super(BCELoss, self).__init__()
        self.prob_min = prob_min
        self.num_classes = num_classes
        self.reduction = reduction
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
       
    def construct(self, x, negative_target, pseudo_target):
        ce_loss = self.cross_entropy(x, pseudo_target)
        # n * k
        prob = ops.softmax(x, axis=-1)
        prob = ops.clip_by_value(prob, self.prob_min, 1.0)
        # n * k
        one_hot = ops.one_hot(negative_target, self.num_classes, Tensor(1.0), Tensor(0.0))
        one_hot = ops.clip_by_value(one_hot, self.prob_min, 1.0)
        sub_prob = ops.clip_by_value(1.0 - prob, self.prob_min, 1.0)
        # n * 1
        nce_loss = -1.0 * ops.sum(one_hot * ops.log(sub_prob), -1)
        
        loss = 0.5 * (ce_loss + nce_loss)
        if self.reduction == "mean":
            loss = ops.mean(loss)
        elif self.reduction == "sum":
            loss = ops.sum(loss)
        return loss

class ORCELoss(nn.Cell):
    def __init__(self, prob_min=1e-7, one_hot_min=1e-2, num_classes=10, reduction="mean"):
        super(ORCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.prob_min = prob_min
        self.one_hot_min = one_hot_min

    def construct(self, x, target):
        prob = ops.softmax(x, axis=-1)
        prob = ops.clip_by_value(1.0 - prob, self.prob_min, 1.0)
        one_hot = ops.one_hot(target, self.num_classes, Tensor(1.0), Tensor(0.0))
        one_hot = ops.clip_by_value(one_hot, self.one_hot_min, 1.0)
        
        loss = -1 * ops.sum(prob * ops.log(one_hot), -1)

        if self.reduction == "mean":
            loss = ops.mean(loss)
        elif self.reduction == "sum":
            loss = ops.sum(loss)

        return loss

class OSCELoss(nn.Cell):
    def __init__(self, alpha=0.1, beta=0.02, one_hot_min=1.0e-2, num_classes=10, reduction="mean"):
        super(OSCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.one_hot_min = one_hot_min
        self.num_classes = num_classes
        self.reduction = reduction
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction=reduction)
        self.orce_loss = ORCELoss(num_classes=num_classes, reduction=reduction, one_hot_min=one_hot_min)

    def construct(self, x, target):
        ce_loss = self.cross_entropy(x, target)
        orce_loss = self.orce_loss(x, target)
        loss = self.alpha * ce_loss + self.beta * orce_loss
        return loss

class DAU(Base, Defense):
    """
    According to the specific defense strategy, override the create_poisoned_dataset() function of the parent class 
        to realize the algorithmic logic of generating the poisoned dataset

    Args:
        task(dict):The defense strategy is used for the task, including datasets, model, Optimizer algorithm 
            and loss function.
        defense_schedule(dict): Parameters are needed according to defense strategy
        schedule=None(dict): Config related to model training
 
    Attributes:
        self.defense_schedule(dict): Initialized by the incoming  parameter "defense_schedule".
        self.defense_strategy(string): The name of defense_strategy.
    """
    def __init__(self, task, defense_schedule):
        schedule = None
        if 'schedule' in defense_schedule:
            schedule = defense_schedule['schedule']
        Base.__init__(self, task, schedule=schedule)   
        Defense.__init__(self)
        
        self.num_classes = defense_schedule["filter"]["num_classes"]

        self.celoss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='sum')
        self.adv_loss = ADVLoss(num_classes=defense_schedule["filter"]["num_classes"], reduction="sum")
        self.bce_loss = BCELoss(num_classes=defense_schedule["filter"]["num_classes"], reduction="sum")
        self.entropy_loss = EntropyLoss(num_classes=defense_schedule["filter"]["num_classes"], reduction="sum")
        self.global_defense_schedule = defense_schedule
       
    def get_poison_data_pool(self):
        return self.poison_data_pool
    
    def get_clean_data_pool(self):
        return self.clean_data_pool

    def get_defense_strategy(self):
        return self.global_defense_schedule["defense_strategy"]
    
    def get_target_label(self):
        pass

    def anti_learning(self, pre_train_model=None, dataset=None, schedule=None):
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
       
        schedule = current_schedule["schedule"]
        schedule['lr'] = current_schedule['anti_learning']['lr']
        schedule['batch_size'] = current_schedule['anti_learning']['batch_size']
        schedule['epochs'] = current_schedule["anti_learning"]["epochs"]
        anti_learning_config = current_schedule["anti_learning"]
        
        if current_schedule['filter']['pre_filter'] is True:
            preserve_indices, suspicious_indices = self.filter(model=deepcopy(pre_train_model), dataset=dataset, schedule=current_schedule, pre_filter=True)
            suspicious_indices = np.setdiff1d(np.arange(len(dataset)), preserve_indices)
            if len(preserve_indices) > 0:
                poison_indices = dataset.get_poison_indices()
                log(f"pre_threshold:{current_schedule['filter']['pre_threshold']}, preserve_indices:{len(preserve_indices)}\n")
                precited = np.zeros(len(dataset))
                print(f"preserve_indices:{preserve_indices}\n")
                precited[preserve_indices] = 1
                expected = np.zeros(len(dataset))
                expected[poison_indices] = 1
                tp, fp, tn, fn = compute_confusion_matrix(precited,expected)
                log(f"preserve_indices:{len(preserve_indices)}, tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}\n")
        else:
            preserve_indices = None
            suspicious_indices = np.arange(len(dataset))
        
        start_time = time.time()

        model = self.get_model()

        self.anti_train(model=model, dataset=dataset, schedule=schedule, anti_learning_config=anti_learning_config)
        
        model = self.get_model()
        poison_data_pool, clean_data_pool = self.filter(model=model, dataset=dataset, schedule=current_schedule, pre_filter=False)

        tp, fp, tn, fn = evaluate_filter(dataset, poison_data_pool)
        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        log(f"poison_data_pool:{len(poison_data_pool)}, tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn},accuracy:{accuracy}, precision:{precision}, recall:{recall}, F1:{F1}\n")
        
        poison_data_pool = np.intersect1d(suspicious_indices, poison_data_pool)
        current_schedule["anti_learning"]["init_poison_data_pool"] = poison_data_pool
        tp, fp, tn, fn = evaluate_filter(dataset, poison_data_pool)
        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        log(f"poison_data_pool:{len(poison_data_pool)}, tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn},accuracy:{accuracy}, precision:{precision}, recall:{recall}, F1:{F1}\n")

        end_time = time.time()
        print(f"Anti_learning end, speed time(s):{end_time - start_time}\n")
        
        return poison_data_pool, clean_data_pool    

    def anti_train(self, model=None, dataset=None, schedule=None, anti_learning_config=None):
        if model is None:
            self.init_model()
        else:
            self.model = deepcopy(model)
      
        train_dataset = dataset
        anti_loss = anti_learning_config["anti_loss"]
        alpha = anti_learning_config["alpha"]
        beta = anti_learning_config["beta"]
        epochs = anti_learning_config['epochs']
        init_poison_data_pool = anti_learning_config["init_poison_data_pool"]
    
        batch_size = schedule['batch_size']

        if len(init_poison_data_pool) < len(train_dataset) / self.num_classes:
            weights = np.ones(len(train_dataset))

            weights[init_poison_data_pool] = (len(train_dataset)/self.num_classes) / len(init_poison_data_pool)
            
            print(f"len(weights):{len(weights)},weights:{weights.shape}\n")
            
            # print(f"weight:{(len(train_dataset)/self.num_classes) / len(init_poison_data_pool)}\n")
            
            sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
            train_loader = GeneratorDataset(
                train_dataset,
                column_names=["image", "label", "index"],
                sampler=sampler,
                num_parallel_workers=1
            )
        else:
            train_loader = GeneratorDataset(
                train_dataset,
                column_names=["image", "label", "index"],
                shuffle=True,
                num_parallel_workers=1
            )
        train_loader = train_loader.batch(
            batch_size=batch_size, 
            drop_remainder=False
        )

        self.model.set_train()
        try:
            classifier_optimizer = self.optimizer(self.model.linear.trainable_params(), learning_rate=schedule['lr'])
        except:
            classifier_optimizer = self.optimizer(self.model.classifier.trainable_params(), learning_rate=schedule['lr'])

        optimizer = classifier_optimizer

        experiment = schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute anti_learning in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)
       
        last_time = time.time()   
        msg = f"Total train samples: {len(train_dataset)}\nInit poisoned data pool:{len(init_poison_data_pool)}\nBatch size: {schedule['batch_size']}\niteration every epoch: {len(train_dataset) // schedule['batch_size']}\nInitial learning rate: {schedule['lr']}\n"
        log(msg)

        # Define train step function
        def forward_fn(batch_img, batch_label, indices_clean, indices_poison):

            predict_digits = self.model(batch_img)
            ce_loss, adv_loss = Tensor(0.0, dtype=ms.float32), Tensor(0.0, dtype=ms.float32)

            indices = indices_clean
            if len(indices) > 0:
                ce_loss = self.celoss(
                    ops.gather(predict_digits, indices, axis=0), 
                    ops.gather(batch_label, indices, axis=0)
                ) * (1.0 / batch_size)

            indices = indices_poison
            if len(indices) > 0:
                if anti_loss == "nceloss":
                    adv_loss = -1.0 * self.celoss(
                        ops.gather(predict_digits, indices, axis=0),
                        ops.gather(batch_label, indices, axis=0)
                    ) * (1.0 / batch_size)
                elif anti_loss == "oceloss":
                    adv_loss = self.adv_loss(
                        ops.gather(predict_digits, indices, axis=0),
                        ops.gather(batch_label, indices, axis=0)
                    ) * (1.0 / batch_size)

            loss = alpha * ce_loss + beta * adv_loss

            return loss, ce_loss, adv_loss, adv_loss, predict_digits
        
        
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        
        def train_step(batch_img, batch_label, indices_clean, indices_poison):
            (loss, ce_loss, adv_loss, adv_loss, predict_digits), grads = grad_fn(batch_img, batch_label, indices_clean, indices_poison)
            optimizer(grads)
            return loss, ce_loss, adv_loss, adv_loss, predict_digits
        
        iteration = 0
        for epoch in range(epochs):
            for batch in train_loader.create_dict_iterator():
                batch_img = batch["image"]
                batch_label = batch["label"]
                batch_indices = batch["index"].asnumpy()

                poison_indices = np.intersect1d(batch_indices, init_poison_data_pool)
                clean_indices = np.setdiff1d(batch_indices, poison_indices)

                indices_clean_np  = np.where(np.isin(batch_indices, clean_indices))[0]
                indices_poison_np = np.where(np.isin(batch_indices, poison_indices))[0]
                indices_clean = ms.Tensor(list(indices_clean_np), dtype=ms.int32)
                indices_poison = ms.Tensor(list(indices_poison_np), dtype=ms.int32)

                # print(f"indices_clean:{len(indices_clean)}, indices_poison:{len(indices_poison)}\n")
                
                loss, ce_loss, adv_loss, adv_loss, predict_digits = train_step(batch_img, batch_label, indices_clean, indices_poison)

                iteration += 1
                if iteration % schedule['log_iteration_interval'] == 0:
                    now = time.time()
                    log(time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +
                        f"Epoch:{epoch+1}/{epochs}, Iteration:{iteration}, lr:{schedule['lr']}, anti_loss:{anti_loss}, loss:{loss}, ce_loss:{ce_loss}, adv_loss:{adv_loss}, time:{now - last_time}\n")

                    pred_class = ops.argmax(predict_digits, dim=1)
                    correct = ops.equal(pred_class, batch_label)
                    correct_num = ops.sum(correct.astype(ms.int32)).asnumpy().item()
                    log(f"batch_size:{batch_size}, correct_num:{correct_num}, poison_indices:{len(poison_indices)}\n")

    def unlearning(self, args=None, model=None, dataset=None, test_dataset=None, schedule=None, result_dict=None):
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        train_dataset = dataset
        unlearning_config = schedule["unlearning"] 
        schedule = current_schedule["schedule"]
        schedule["lr"] = unlearning_config["lr"]
        schedule["batch_size"] = unlearning_config["batch_size"]
        schedule["gamma"] = unlearning_config["gamma"]
        schedule["schedule"] = unlearning_config["schedule"]
        schedule["lr_scheduler_config"] = unlearning_config["lr_scheduler_config"]

        start_time = time.time()
        test_results = self.unlearning_train(
            args = args, 
            model=model, 
            dataset=train_dataset,
            test_dataset=test_dataset, 
            schedule=schedule, 
            unlearning_config=unlearning_config, 
            result_dict=result_dict
        )

        end_time = time.time()
        print(f"Unlearning ends, speed time(s):{end_time - start_time}\n")

        return test_results
    
    def unlearning_train(self,  args=None, model=None, dataset=None, test_dataset=None, schedule=None, unlearning_config=None, result_dict=None):
        
        result_save_path = args['save_path']
        if model is None:
            self.init_model()
        else:
            self.model = model
        
        tau = unlearning_config["tau"]

        alpha = args['custom_parameter']['alpha']
        beta = args['custom_parameter']['beta']
        epochs = args['custom_parameter']['epochs']

        # alpha = unlearning_config["alpha"]
        # beta = unlearning_config["beta"]
        # epochs = unlearning_config['epochs']

        unlearning_config["poison_data_pool"]

        log(f"alpha:{alpha},beta:{beta},epochs:{epochs}\n")

        poison_data_pool = unlearning_config["poison_data_pool"]
        clean_data_pool = unlearning_config["clean_data_pool"]
        unlearning_loss = unlearning_config["unlearning_loss"]

        train_dataset = dataset
        batch_size = schedule['batch_size'] 
        train_loader = mindspore.dataset.GeneratorDataset(
            train_dataset,
            column_names=["image", "label", "index"],
            shuffle=True,
            num_parallel_workers=1
        ).batch(batch_size=batch_size, drop_remainder=False)

        self.model.set_train()
        optimizer = nn.SGD(self.model.trainable_params(), learning_rate=schedule['lr'], momentum=schedule['momentum'], weight_decay=schedule['weight_decay'])                                    
        
        experiment = schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute model train in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)
        last_time = time.time()   
        msg = f"Total train samples: {len(train_dataset)}\nPoison data pool:{len(poison_data_pool)}\nBatch size: {schedule['batch_size']}\niteration every epoch: {len(train_dataset) // schedule['batch_size']}\nInitial learning rate: {schedule['lr']}\n"
        log(msg)


        test_results = {"acc":None, "asr":None}
        loss_list, acc_list, asr_list = [], [], []
        iteration = 0

        def forward_fn(img, label):
            logits = self.model(img)
            ce_loss = 0.0
            cce_loss = 0.0
            nce_loss = 0.0
            loss = 0.0

            clean_mask = np.isin(batch_indices, clean_data_pool)
            poison_mask = np.isin(batch_indices, poison_data_pool)

            clean_idx = np.where(clean_mask)[0]
            poison_idx = np.where(poison_mask)[0]

            if len(clean_idx) > 0:    

                clean_pred = ops.gather(logits, Tensor(clean_idx, ms.int32), 0)
                clean_lbl = ops.gather(label, Tensor(clean_idx, ms.int32), 0)
            
                ce_loss = self.celoss(clean_pred, clean_lbl) / batch_size
                loss += alpha * ce_loss

            if len(poison_idx) > 0:

                # print(f"poison_idx:{len(poison_idx)}\n")
                
                poison_pred = ops.gather(logits, Tensor(poison_idx, ms.int32), 0)
                poison_lbl = ops.gather(label, Tensor(poison_idx, ms.int32), 0)
            
                if unlearning_loss == "nceloss":
                    nce_loss = -self.celoss(poison_pred, poison_lbl) / batch_size
                    loss += beta * nce_loss

                elif unlearning_loss == "cceloss":
                    cce_loss = self.adv_loss(poison_pred, poison_lbl) / batch_size
                    loss += beta * cce_loss

                elif unlearning_loss == "both":
                    nce_loss = -self.celoss(poison_pred, poison_lbl) / batch_size
                    cce_loss = self.adv_loss(poison_pred, poison_lbl) / batch_size
                    loss += beta * cce_loss + tau * nce_loss

                # print(f"loss:{loss}, ce_loss:{ce_loss}, cce_loss:{cce_loss}, nce_loss:{nce_loss}\n")
                
            return loss, ce_loss, cce_loss, nce_loss
        
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # poisoned_test_indexs = test_dataset.get_poison_indices()
        # print(f"poisoned_test_indexs:{len(poisoned_test_indexs)}\n")

        result_dict = {}
        for i in range(epochs):
            for batch in train_loader.create_dict_iterator():
                batch_img = batch["image"]
                batch_label = batch["label"]
                batch_indices = batch["index"].asnumpy()

                (loss_val, ce_loss, cce_loss, nce_loss), grads = grad_fn(batch_img, batch_label)

                optimizer(grads)

                iteration += 1

                log_iteration_interval = schedule['log_iteration_interval']
                log_iteration_interval = 100
                if iteration % log_iteration_interval == 0:
                    pred = self.model(batch_img)
                    pred_label = ops.Argmax()(pred)
                    correct = ops.Equal()(pred_label, batch_label)
                    correct_count = ops.sum(correct.astype(ms.int32))  #

                    log(f"Epoch:{i+1}, Iter:{iteration}, loss:{loss_val}, ce:{ce_loss}, cce:{cce_loss}, nce:{nce_loss}, correct_count:{correct_count}\n")
            
            test_iteration_interval = 1
            if test_dataset is not None and (i + 1) % test_iteration_interval == 0:
                poisoned_test_indexs = test_dataset.get_poison_indices()
                benign_test_indexs = list(set(range(len(test_dataset))) - set(poisoned_test_indexs))
                
                if not isinstance(poisoned_test_indexs, Tensor):
                    poisoned_test_indexs = Tensor(poisoned_test_indexs, dtype=ms.int32)

                print(f"type(poisoned_test_indexs):{type(poisoned_test_indexs)}\n")
               
                # poisoned_test_indexs = poisoned_test_indexs.tolist()

                pred_logits, labels = self._test(test_dataset, model=self.model)
                acc = compute_accuracy(pred_logits[benign_test_indexs], labels[benign_test_indexs], topk=(1, 3, 5))
                asr = compute_accuracy(pred_logits[poisoned_test_indexs], labels[poisoned_test_indexs], topk=(1, 3, 5))

                loss_list.append(loss_val.asnumpy().tolist())
                acc_list.append(acc[0].asnumpy().tolist())
                asr_list.append(asr[0].asnumpy().tolist())

                result_dict["loss"] = loss_list
                result_dict["asr"] = asr_list
                result_dict["acc"] = acc_list

                # print(f"loss:{loss_list}\n")
                # print(f"asr:{asr_list}\n")
                # print(f"acc:{acc_list}\n")
                
                with open(result_save_path, 'w') as f:
                    json.dump(result_dict, f)

        result_dict["status"] = "completed"
        with open(result_save_path, 'w') as f:
            json.dump(result_dict, f)

        test_results["acc"] = np.array(acc)
        test_results["asr"] = np.array(asr)
        return test_results


    # def unlearning(self, args, model=None, dataset=None, test_dataset=None, schedule=None, result_dict=None):
    #     if schedule is not None:
    #         current_schedule = deepcopy(schedule)
    #     elif self.global_defense_schedule is not None:
    #         current_schedule = deepcopy(self.global_defense_schedule)
    #     else: 
    #         raise AttributeError("Training schedule is None, please check your schedule setting.")
    #     train_dataset = dataset
    #     unlearning_config = schedule["unlearning"] 
    #     schedule = current_schedule["schedule"]
    #     schedule["lr"] = unlearning_config["lr"]
    #     schedule["batch_size"] = unlearning_config["batch_size"]
    #     schedule["gamma"] = unlearning_config["gamma"]
    #     schedule["schedule"] = unlearning_config["schedule"]
    #     schedule["lr_scheduler_config"] = unlearning_config["lr_scheduler_config"]

    #     start_time = time.time()
    #     test_results = self.unlearning_train(args, model=model, dataset=train_dataset, test_dataset=test_dataset, schedule=schedule, unlearning_config=unlearning_config, result_dict=result_dict)
    #     end_time = time.time()
    #     print(f"Unlearning ends, speed time(s):{end_time - start_time}\n")

    #     return test_results
    
    # def unlearning_train(self, args, model=None, dataset=None, test_dataset=None, schedule=None, unlearning_config=None, result_dict=None):
    #     result_save_path = args['save_path']

    #     if model is None:
    #         self.init_model()
    #     else:
    #         self.model = model
        
    #     tau = unlearning_config["tau"]
    #     alpha = args["custom_parameter"]["alpha"]
    #     beta = args["custom_parameter"]["beta"]
    #     epochs = args["custom_parameter"]['epochs']

    #     log(f"alpha:{alpha},beta:{beta},epochs:{epochs}\n")

    #     poison_data_pool = unlearning_config["poison_data_pool"]
    #     clean_data_pool = unlearning_config["clean_data_pool"]
    #     unlearning_loss = unlearning_config["unlearning_loss"]

    #     train_dataset = dataset
    #     batch_size = schedule['batch_size'] 
    #     train_loader = mindspore.dataset.GeneratorDataset(
    #         train_dataset,
    #         column_names=["data", "label", "index"],
    #         shuffle=True,
    #         num_parallel_workers=1
    #     )
    #     train_loader = train_loader.batch(batch_size=batch_size, drop_remainder=False)

    #     self.model.set_train()
    #     optimizer = nn.SGD(self.model.trainable_params(), learning_rate=schedule['lr'], momentum=schedule['momentum'], weight_decay=schedule['weight_decay'])                                    
        
 
    #     experiment = schedule['experiment']
    #     t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    #     msg = "\n==========Execute model train in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
    #     log(msg)
    #     last_time = time.time()   
    #     msg = f"Total train samples: {len(train_dataset)}\nPoison data pool:{len(poison_data_pool)}\nBatch size: {schedule['batch_size']}\niteration every epoch: {len(train_dataset) // schedule['batch_size']}\nInitial learning rate: {schedule['lr']}\n"
    #     log(msg)


    #     test_results = {"acc":None, "asr":None}
    #     loss_list, acc_list, asr_list = [], [], []
    #     iteration = 0

    #     for i in range(epochs):
    #         for batch in train_dataset.create_dict_iterator():
    #             batch_img = batch["image"]
    #             batch_label = batch["label"]
    #             batch_indices = batch["index"].asnumpy()

    #             def forward_fn(img, label):
    #                 logits = self.model(img)
    #                 ce_loss = 0.0
    #                 cce_loss = 0.0
    #                 nce_loss = 0.0
    #                 loss = 0.0

    #                 clean_mask = np.isin(batch_indices, clean_data_pool)
    #                 poison_mask = np.isin(batch_indices, poison_data_pool)

    #                 clean_idx = np.where(clean_mask)[0]
    #                 poison_idx = np.where(poison_mask)[0]

    #                 if len(clean_idx) > 0:
    #                     clean_pred = ops.gather(logits, Tensor(clean_idx, ms.int32), 0)
    #                     clean_lbl = ops.gather(label, Tensor(clean_idx, ms.int32), 0)
    #                     ce_loss = self.celoss(clean_pred, clean_lbl) / batch_size
    #                     loss += alpha * ce_loss

    #                 if len(poison_idx) > 0:
    #                     poison_pred = ops.gather(logits, Tensor(poison_idx, ms.int32), 0)
    #                     poison_lbl = ops.gather(label, Tensor(poison_idx, ms.int32), 0)
    #                     if unlearning_loss == "nceloss":
    #                         nce_loss = -self.celoss(poison_pred, poison_lbl) / batch_size
    #                         loss += beta * nce_loss
    #                     elif unlearning_loss == "cceloss":
    #                         cce_loss = self.adv_loss(poison_pred, poison_lbl) / batch_size
    #                         loss += beta * cce_loss
    #                     elif unlearning_loss == "both":
    #                         nce_loss = -self.celoss(poison_pred, poison_lbl) / batch_size
    #                         cce_loss = self.adv_loss(poison_pred, poison_lbl) / batch_size
    #                         loss += beta * cce_loss + tau * nce_loss

    #                 return loss, ce_loss, cce_loss, nce_loss

    #             grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    #             (loss_val, ce_loss, cce_loss, nce_loss), grads = grad_fn(batch_img, batch_label)
    #             optimizer(grads)

    #             iteration += 1
    #             if iteration % schedule['log_iteration_interval'] == 0:
    #                 pred = self.model(batch_img)
    #                 pred_label = ops.Argmax()(pred)
    #                 correct = ops.ReduceSum()(ops.Equal()(pred_label, batch_label))
    #                 log(f"Epoch:{i+1}, Iter:{iteration}, loss:{loss_val}, ce:{ce_loss}, cce:{cce_loss}, nce:{nce_loss}, correct:{correct}\n")
            
    #         if test_dataset is not None:
    #             poisoned_test_indexs = test_dataset.get_poison_indices()
    #             benign_test_indexs = list(set(range(len(test_dataset))) - set(poisoned_test_indexs))

    #             pred_logits, labels = self._test(test_dataset, model=self.model)
    #             acc = compute_accuracy(pred_logits[benign_test_indexs], labels[benign_test_indexs], topk=(1, 3, 5))
    #             asr = compute_accuracy(pred_logits[poisoned_test_indexs], labels[poisoned_test_indexs], topk=(1, 3, 5))

    #             loss_list.append(loss_val.asnumpy().tolist())
    #             acc_list.append(acc[0].asnumpy().tolist())
    #             asr_list.append(asr[0].asnumpy().tolist())

    #             result_dict["loss"] = loss_list
    #             result_dict["asr"] = asr_list
    #             result_dict["acc"] = acc_list

    #             with open(result_save_path, 'w') as f:
    #                 json.dump(result_dict, f)

    #     result_dict["status"] = "completed"
    #     with open(result_save_path, 'w') as f:
    #         json.dump(result_dict, f)

    #     test_results["acc"] = np.array(acc)
    #     test_results["asr"] = np.array(asr)
    #     return test_results

    
    def repair(self, model=None, dataset=None, schedule=None):
        pass
   
    def filter(self, model=None, dataset=None, schedule=None, pre_filter=False):
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
           
        if model is None:
            pre_train_model = schedule["anti_learning"]["pre_train_model"] 
            self.anti_learning(pre_train_model=pre_train_model, dataset=dataset, schedule=schedule)
            model = self.get_model()

        pre_threshold = current_schedule["filter"]["pre_threshold"]
        threshold = current_schedule["filter"]["threshold"]
        top_k = current_schedule["filter"]["top_k"]
        
        if pre_filter is True:
            poison_data_pool, clean_data_pool = self._filter(model=model, dataset=dataset, top_k=None, threshold=pre_threshold, defense_schedule=current_schedule) 
        else:
            poison_data_pool, clean_data_pool = self._filter(model=model, dataset=dataset, top_k=top_k, threshold=threshold, defense_schedule=current_schedule)   
        
        return poison_data_pool, clean_data_pool
 

    def _filter(self, model=None, dataset=None, top_k=None, threshold=None, defense_schedule=None):
        alpha = defense_schedule["filter"]["sce"]["alpha"]
        beta = defense_schedule["filter"]["sce"]["beta"]
        one_hot_min = defense_schedule["filter"]["sce"]["one_hot_min"] 
        labels = dataset.get_modified_targets()
        cur_indices = np.arange(len(dataset))
        
        if model is None:
            model = deepcopy(self.model)
           
        model.set_train(False)
        msg = "==========Start filtering suspicious poison samples==========\n"
        log(msg)

        predict_digits, labels = self._get_model_predict(model=model, dataset=dataset)
        sce = core.SCELoss(alpha=alpha, beta=beta, one_hot_min=one_hot_min, num_classes=self.num_classes, reduction="none")
        sceloss = sce(predict_digits, labels)
        sceloss = sceloss.asnumpy()

        if top_k is not None:
            sorted_indices = np.argsort(sceloss)
            top_indices = sorted_indices[-1*top_k:]
        elif threshold is not None:
            top_indices = np.where(sceloss > threshold)[0]

        poison_data_pool = np.array([cur_indices[index] for index in top_indices])
        clean_data_pool = np.setdiff1d(np.arange(len(dataset)), poison_data_pool)
        return poison_data_pool, clean_data_pool
    
    def _get_model_predict(self, model=None, dataset=None):

        test_loader = mindspore.dataset.GeneratorDataset(
            dataset,
            column_names=["data", "label","index"],
            shuffle=False,
            num_parallel_workers=1
        )
        test_loader = test_loader.batch(batch_size=64, drop_remainder=False)

        predict_digits = []
        labels = []
       
        for batch in test_loader.create_tuple_iterator():
            batch_img, batch_label = batch[0], batch[1]
            batch_img = batch_img
            batch_pred = model(batch_img)
            batch_pred = batch_pred.asnumpy()
            predict_digits.append(batch_pred)
            labels.append(batch_label.asnumpy())
           
        predict_digits = np.concatenate(predict_digits, axis=0)
        labels = np.concatenate(labels, axis=0)

        return Tensor(predict_digits), Tensor(labels)
    
    def _test(self, dataset, model=None, batch_size=16, num_workers=8):
        if model is None:
            model = self.model
        else:
            model = model

        test_loader = GeneratorDataset(
            dataset,
            column_names=["data", "label","index"],
            shuffle=False,
            num_parallel_workers=1
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