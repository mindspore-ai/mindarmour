# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/19 15:49:57
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_attack.py
# @Description : This is the test code of BadNets.              
             
import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# print(BASE_DIR)

# MindSpore imports
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.common.initializer import Normal
from mindspore.dataset import vision, transforms
from mindspore.dataset import Cifar10Dataset, MnistDataset, ImageFolderDataset

# core
import core
from core.attacks import BackdoorAttack
# from core.attacks import BadNets
from core.attacks import BadNets, Blended, Refool, SIG
# from core.attacks import BadNets, Blended, WaNet, Refool
from config import get_task_config, get_task_schedule, get_attack_config

# numpy
import pickle
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import manifold

# utils
from utils import Log, parser, compute_accuracy, read_image, save_img


# ==================== Set global settings ====================
args = parser.parse_args()
dataset = args.dataset

# # Set context (GPU/CPU)
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# ['Attack', 'BackdoorAttack', 'BadNets', 'Blended', 'IAD', 'WaNet', 'SIG', 'Refool']

attack = 'Refool'
attack_schedule = get_attack_config(attack_strategy=attack, dataset=dataset)

poisoned_rate = attack_schedule["poisoned_rate"]
y_target = attack_schedule["y_target"]

experiment_dir = os.path.join(BASE_DIR, "save/")

if dataset == "CIFAR-10":
    experiment = f'ResNet-18_CIFAR-10_{attack}' 
    task = 'ResNet-18_CIFAR-10'
    layer = "linear"
    num_classes = 10
   
elif dataset == "GTSRB":
    num = 19
    experiment = f'ResNet-18_GTSRB_{attack}' 
    task = 'ResNet-18_GTSRB'
    layer = "linear"
    num_classes = 43

work_dir = os.path.join(experiment_dir, f'{task}/attack/{attack}/')

# dataset
datasets_dir = os.path.join(work_dir,'datasets')
poison_datasets_dir = os.path.join(datasets_dir, 'poisoned_data')
poison_datasets_dir = os.path.join(poison_datasets_dir, f"y_target_{y_target}_poison_{poisoned_rate}/")
poison_train_datasets_file = 'poisoned_train_dataset.npz'
poison_test_datasets_file = 'poisoned_test_dataset.npz'
# model 
clean_model_dir = os.path.join(experiment_dir, f'{task}/clean_model/')
clean_model_file = "clean_model.ckpt"

model_dir = os.path.join(work_dir,'model')
model_dir = os.path.join(model_dir, f"y_target_{y_target}_poison_{poisoned_rate}/")

ckpt_dir = os.path.join(model_dir, f"ckpt/")
show_dir = os.path.join(work_dir, f"show/y_target_{y_target}_poison_{poisoned_rate}/")

dirs = [work_dir, datasets_dir, poison_datasets_dir, model_dir, ckpt_dir, show_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# ============================== Load global config ==============================

task_config = get_task_config(task = task)
schedule = get_task_schedule(task = task)
schedule["ckpt_dir"] = ckpt_dir
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir

attack_schedule['work_dir'] = work_dir
attack_schedule['train_schedule'] = schedule

if task_config["model_type"] == "Decoupled":
    model_dir = os.path.join(model_dir,"decoupled_model/") 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(model_dir)

def get_attack_method(attack, task_config, attack_schedule):
    global save_dataset, load_dataset
    if attack == "BadNets":
        attack_method = BadNets(
            task_config,
            attack_schedule
        )
        from core.attacks.BadNets import save_dataset as badnets_save
        from core.attacks.BadNets import load_dataset as badnets_load
        save_dataset, load_dataset = badnets_save, badnets_load

    elif attack == "Blended":
        attack_method = Blended(
            task_config,
            attack_schedule
        )
        from core.attacks.Blended import save_dataset as blended_save
        from core.attacks.Blended import load_dataset as blended_load
        save_dataset, load_dataset = blended_save, blended_load

    elif attack == "Refool":
        # load reflection images
        reflection_images = []
        reflection_data_dir = os.path.join(BASE_DIR,"datasets/VOCdevkit/VOC2012/JPEGImages/")
        reflection_image_path = os.listdir(reflection_data_dir)
        reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]

        print(f"reflection_images:{reflection_images[0].shape}")

        attack_schedule['reflection_candidates'] = reflection_images
        attack_method = Refool(
            task_config,
            attack_schedule
        )
        from core.attacks.Refool import save_dataset as refool_save
        from core.attacks.Refool import load_dataset as refool_load
        save_dataset, load_dataset = refool_save, refool_load

    elif attack == "SIG":

        attack_method = SIG(
            task_config,
            attack_schedule
        )
        from core.attacks.SIG import save_dataset as sig_save
        from core.attacks.SIG import load_dataset as sig_load
        save_dataset, load_dataset = sig_save, sig_load

    return attack_method

if __name__ == "__main__":

    """
    Users can select the task to execute by passing parameters.
    1. The task of generating and showing backdoor samples

        python test_attack.py --dataset "CIFAR-10"  --subtask "generate backdoor samples"
        python test_attack.py --dataset "GTSRB"  --subtask "generate backdoor samples"  

        python test_attack.py --dataset "CIFAR-10" --subtask "show backdoor samples"
        python test_attack.py --dataset "GTSRB" --subtask "show backdoor samples"

    2. The task of training backdoor model

        python test_attack.py --dataset "CIFAR-10" --subtask "clean_train"
        python test_attack.py --dataset "GTSRB" --subtask "clean_train"

        python test_attack.py --dataset "CIFAR-10" --subtask "attack"
        python test_attack.py --dataset "GTSRB" --subtask "attack"
        
    3.The task of testing backdoor model
     
        python test_attack.py --dataset "CIFAR-10" --subtask "test"
        python test_attack.py --dataset "GTSRB" --subtask "test"

    """

    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)

    attack_method = get_attack_method(attack, task_config, attack_schedule)
    backdoor = BackdoorAttack(attack_method)
 
    if args.subtask == "generate backdoor samples":
        # Generate backdoor sample
        log("\n==========Generate backdoor samples==========\n")
        poisoned_train_dataset = backdoor.create_poisoned_train_dataset()
        poison_indices = poisoned_train_dataset.get_poison_indices()
        benign_indexs = list(set(range(len(poisoned_train_dataset))) - set(list(poison_indices)))
        
        # Statistically generated poisoned datasets information
        log("\n==========Statistically generated poisoned datasets information==========\n")
        real_targets = np.array(poisoned_train_dataset.get_real_targets())
        labels = real_targets[poison_indices]
        log(f"Total samples:{len(poisoned_train_dataset)}, poisoning samples:{len(poison_indices)}, benign samples:{len(benign_indexs)}\n")
        for i, label in enumerate(poisoned_train_dataset.classes):
            print(f"the number of sample with label:{label} in poisoned_train_dataset:{labels.tolist().count(i)}\n")
        
        # Save poisoned dataset
        save_dataset(poisoned_train_dataset, os.path.join(poison_datasets_dir, poison_train_datasets_file))
        log("Save generated train_datasets to" + os.path.join(poison_datasets_dir,poison_train_datasets_file))
      
        poisoned_test_dataset = backdoor.create_poisoned_test_dataset(poisoned_rate=0.1)
        real_targets = np.array(poisoned_test_dataset.get_real_targets())
        poison_indices = poisoned_test_dataset.get_poison_indices()
        labels = real_targets[poison_indices]
        poison_test_indices = poisoned_test_dataset.get_poison_indices()
        benign_test_indexs = list(set(range(len(poisoned_test_dataset))) - set(poison_test_indices))

        log("Total samples:{0}, poisoning samples:{1}, benign samples:{2}".format(len(poisoned_test_dataset),len(poison_test_indices),len(benign_test_indexs)))
                                                                                  
        for i, label in enumerate(poisoned_test_dataset.classes):
            print(f"the number of sample with label:{label} in poisoned_train_dataset:{labels.tolist().count(i)}\n")
      
        save_dataset(poisoned_test_dataset, os.path.join(poison_datasets_dir, poison_test_datasets_file))
        log("Save generated train_datasets to" + os.path.join(poison_datasets_dir,poison_test_datasets_file))


    elif args.subtask == "show backdoor samples":
        log("\n==========Show poisoning train sample==========\n")
 
        poisoned_train_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_train_datasets_file))
        poisoned_test_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_test_datasets_file))

        poison_indices = poisoned_train_dataset.get_poison_indices()

        index = poison_indices[random.choice(range(len(poison_indices)))] 

        log(f"Random index:{index}") 

        image, label, _ = poisoned_train_dataset[index]
        if isinstance(image, mindspore.Tensor):
            image = image.asnumpy()

        print(f"index:{index},image:{type(image)}\n")

        backdoor_sample_path = os.path.join(show_dir, f"backdoor_train_sample_index_{index}.png")
       
        title = f"Target label:{label}" 

        save_img(image, title=title, path=backdoor_sample_path)
        log("Save train backdoor samples to" + backdoor_sample_path)

        log("\n==========Show train sample==========\n")
        
        train_dataset = task_config["train_dataset"]
        train_list = list(train_dataset.create_tuple_iterator())
        image, label = train_list[index]
        if isinstance(image, mindspore.Tensor):
            image = image.asnumpy()
        
        # print(f"image:{image}\n")

        sample_path = os.path.join(show_dir, f"train_sample_index_{index}.png")
        title = f"Origin label:{label}" 
        save_img(image, title=title, path=sample_path)
        log("Save train samples to" + sample_path)

    elif args.subtask == "attack":
        #Train and get backdoor model
        log("\n==========Train on poisoned_train_dataset and get backdoor model==========\n")
        
        # Load datasets in MindSpore way

        poisoned_train_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_train_datasets_file))
        poisoned_test_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_test_datasets_file))

        backdoor.attack(train_dataset=poisoned_train_dataset, test_dataset=poisoned_test_dataset)
        poisoned_model = backdoor.get_backdoor_model()

        # Save model in MindSpore way
        mindspore.save_checkpoint(poisoned_model, os.path.join(model_dir, f'backdoor_model.ckpt'))
        log("Save backdoor model to" + os.path.join(model_dir, f'backdoor_model.ckpt'))

    elif args.subtask == "test":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of backdoor attack on poisoned_test_dataset==========\n")
           
        poisoned_test_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_test_datasets_file))

        # Load dataset in MindSpore way
        # poisoned_test_dataset = mindspore.dataset.GeneratorDataset(poisoned_test_dataset, ['image', 'label', 'poisoned'])
        
        testset = poisoned_test_dataset
        poisoned_test_indexs = testset.get_poison_indices()
        poisoned_test_indexs = poisoned_test_indexs.tolist()
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
       
        # Load model in MindSpore way
        print(os.path.join(model_dir, f'backdoor_model.ckpt'))
        model = task_config['model']
        param_dict = mindspore.load_checkpoint(os.path.join(model_dir, f'backdoor_model.ckpt'))
        mindspore.load_param_into_net(model, param_dict)
     
        predict_digits, labels = backdoor.test(model=model, test_dataset=testset)
        benign_acc = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs])
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs])
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(testset),len(poisoned_test_indexs), len(benign_test_indexs)))                                                                                                                                                
        log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc[0], poisoned_acc[0]))