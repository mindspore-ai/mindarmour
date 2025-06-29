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
# paddle
import paddle
# core
import core
from core.attacks import BackdoorAttack
# from core.attacks import BadNets, Blended, IAD, WaNet, Refool
from core.attacks import BadNets, Blended, WaNet, Refool
from config import get_task_config, get_task_schedule, get_attack_config
# numpy
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import manifold

# utils
from implementations.training_tamper.mindspore_impl.DAU.utils import Log, parser
from implementations.training_tamper.mindspore_impl.DAU.utils import save_img
from implementations.training_tamper.mindspore_impl.DAU.utils import compute_accuracy
from implementations.training_tamper.mindspore_impl.DAU.utils import read_image

# ==================== Set global settings ====================
args = parser.parse_args()
dataset = args.dataset

# ['Attack', 'BackdoorAttack', 'BadNets', 'Blended', 'IAD', 'WaNet', 'Refool']
attack = 'BadNets'
attack_schedule = get_attack_config(attack_strategy=attack, dataset=dataset)

poisoned_rate = attack_schedule["poisoned_rate"]
y_target = attack_schedule["y_target"]

experiment_dir = os.path.join(BASE_DIR, "save/")
if dataset == "CIFAR-10":
    #{model}_{datasets}_{attack}_{defense}
    experiment = f'ResNet-18_CIFAR-10_{attack}' 
    task = 'ResNet-18_CIFAR-10'
    layer = "linear"
    num_classes = 10
   
elif dataset == "GTSRB":
    # ========== ResNet-18_CIFAR-100 ==========
    num = 19
    # experiment = f'VGG{num}_GTSRB'
    # task = f'VGG_GTSRB'
    experiment = f'ResNet-18_GTSRB_{attack}' 
    task = 'ResNet-18_GTSRB'
    layer = "linear"
    num_classes = 43

work_dir = os.path.join(experiment_dir, f'{task}/attack/{attack}/')

# dataset
datasets_dir = os.path.join(work_dir,'datasets')
poison_datasets_dir = os.path.join(datasets_dir, 'poisoned_data')
poison_datasets_dir = os.path.join(poison_datasets_dir, f"y_target_{y_target}_poison_{poisoned_rate}/")

# model 
clean_model_dir = os.path.join(experiment_dir, f'{task}/clean_model/')
clean_model_file = "clean_model.pth"

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
    if attack == "BadNets":
        attack_method = BadNets(
            task_config,
            attack_schedule
        )
    # elif attack == "Blended":
    #     attack_method = Blended(
    #         task_config,
    #         attack_schedule
    #     )
    # elif attack == "WaNet":
    #     attack_method = WaNet(
    #         task_config,
    #         attack_schedule
    #     )
    # elif attack == "Refool":
    #     # load reflection images
    #     reflection_images = []
    #     reflection_data_dir = os.path.join(BASE_DIR,"datasets/VOCdevkit/VOC2012/JPEGImages/")
    #     reflection_image_path = os.listdir(reflection_data_dir)
    #     reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
    #     attack_schedule['reflection_candidates'] = reflection_images
    #     attack_method = Refool(
    #         task_config,
    #         attack_schedule
    #     )
    return attack_method

if __name__ == "__main__":
    """
    Users can select the task to execute by passing parameters.
    1. The task of generating and showing backdoor samples
      
        python test_attack.py --dataset "CIFAR-10"  --subtask "generate backdoor samples"
        python test_attack.py --dataset "GTSRB"  --subtask "generate backdoor samples"  
        python test_attack.py --dataset "Tiny-ImageNet"  --subtask "generate backdoor samples"  

        python test_attack.py --dataset "CIFAR-10" --subtask "show backdoor samples"
        python test_attack.py --dataset "GTSRB" --subtask "show backdoor samples"
        python test_attack.py --dataset "Tiny-ImageNet" --subtask "show backdoor samples"  

    2. The task of training backdoor model

        python test_attack.py --dataset "CIFAR-10" --subtask "clean_train"
        python test_attack.py --dataset "GTSRB" --subtask "clean_train"
        python test_attack.py --dataset "Tiny-ImageNet" --subtask "clean_train" 

      
        python test_attack.py --dataset "CIFAR-10" --subtask "attack"
        python test_attack.py --dataset "GTSRB" --subtask "attack"
        python test_attack.py --dataset "Tiny-ImageNet" --subtask "attack" 

    3.The task of testing backdoor model
     
        python test_attack.py --dataset "CIFAR-10" --subtask "test"
        python test_attack.py --dataset "GTSRB" --subtask "test"
        python test_attack.py --dataset "Tiny-ImageNet" --subtask "test"
    
    """
    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)

    # log(str(task_config['model']))

    # badnets = BadNets(
    #     task_config,
    #     attack_schedule
    # )

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
        print(os.path.join(poison_datasets_dir,'train.pdparams'))

        paddle.save(poisoned_train_dataset, os.path.join(poison_datasets_dir,'train.pdparams'))
        log("Save generated train_datasets to" + os.path.join(poison_datasets_dir,'train.pdparams'))

        poisoned_test_dataset = backdoor.create_poisoned_test_dataset(poisoned_rate=0.1)

        # print(f"poisoned_test_dataset:{len(poisoned_test_dataset)}\n")

        poison_test_indices =  poisoned_test_dataset.get_poison_indices()
        benign_test_indexs = list(set(range(len(poisoned_test_dataset))) - set(poison_test_indices))
        log("Total samples:{0}, poisoning samples:{1}, benign samples:{2}".format(len(poisoned_test_dataset),\
        len(poison_test_indices),len(benign_test_indexs)))
        paddle.save(poisoned_test_dataset, os.path.join(poison_datasets_dir,'test.pdparams'))
        log("Save generated test_datasets to" + os.path.join(poison_datasets_dir,'test.pdparams'))

    elif args.subtask == "show backdoor samples":

        log("\n==========Show posioning train sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_train_dataset = paddle.load(os.path.join(poison_datasets_dir,'train.pdparams')) 
        poison_indices = poisoned_train_dataset.get_poison_indices()

        # Outside of neural networks, packages including numpy and matplotlib are usually used for data operations, 
        # so the type of data is usually converted to np.ndrray()
      
        index = poison_indices[random.choice(range(len(poison_indices)))] 
        log(f"Random index:{index}") 

        image, label, _ = poisoned_train_dataset.get_sample_by_index(index)
        if isinstance(image, paddle.Tensor):
            image = image.numpy()

        # print(f"image:{image}\n")
        backdoor_sample_path = os.path.join(show_dir, f"backdoor_train_sample_index_{index}.png")
       
        title = f"Target label:{label}" 
        save_img(image, title=title, path=backdoor_sample_path)
        log("Save train backdoor samples to" + backdoor_sample_path)

        log("\n==========Show train sample==========\n")
        
        train_dataset = task_config["train_dataset"]
        image, label = train_dataset[index]
        if isinstance(image, paddle.Tensor):
            image = image.numpy()

        # print(f"image:{image}\n")
        sample_path = os.path.join(show_dir, f"train_sample_index_{index}.png")
        title = f"Origin label:{label}." 
        save_img(image, title=title, path=sample_path)
        log("Save train samples to" + sample_path)


        # log("\n==========Show posioning test sample==========\n")
        # # Alreadly exsiting dataset and trained model.
        # poisoned_test_dataset = paddle.load(os.path.join(poison_datasets_dir,'test.pdparams'))
        # poison_indices = poisoned_test_dataset.get_poison_indices()
        # index = poison_indices[random.choice(range(len(poison_indices)))]
        # log(f"Random index:{index}") 

        # # image, label, _ = poisoned_test_dataset[index] 
        # image, label, _ = poisoned_test_dataset.get_sample_by_index(index)
        # image = image.numpy()
        # backdoor_sample_path = os.path.join(show_dir, f"backdoor_test_sample_index_{index}.png")
        # title = f"Target label:{label},{classes[label]}."
        # save_img(image, title=title, path=backdoor_sample_path)
        # log("Save test backdoor samples to" + backdoor_sample_path)

        # test_dataset = task_config["test_dataset"]
        # image, label = test_dataset[index]
        # if isinstance(image, paddle.Tensor):
        #     image = image.numpy()
        # # print(f"image:{image}\n")
        # sample_path = os.path.join(show_dir, f"test_sample_index_{index}.png")
        # title = f"Origin label:{label},{classes[label]}."
        # save_img(image, title=title, path=sample_path)
        # log("Save train backdoor samples to" + sample_path)

    elif args.subtask == "attack":
        #Train and get backdoor model
        log("\n==========Train on poisoned_train_dataset and get backdoor model==========\n")
        
        poisoned_train_dataset = paddle.load(os.path.join(poison_datasets_dir,'train.pdparams')) 
        poisoned_test_dataset = paddle.load(os.path.join(poison_datasets_dir,'test.pdparams')) 

        # print(f"poisoned_train_dataset:{poisoned_train_dataset.transform}\n")
        # print(f"poisoned_test_dataset:{poisoned_test_dataset.transform}\n")

        backdoor.attack(train_dataset=poisoned_train_dataset, test_dataset=poisoned_test_dataset)
        poisoned_model = backdoor.get_backdoor_model()

        paddle.save(poisoned_model.state_dict(), os.path.join(model_dir, f'backdoor_model.pdparams'))
        log("Save backdoor model to" + os.path.join(model_dir, f'backdoor_model.pdparams'))

    
    elif args.subtask == "test":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of backdoor attack on poisoned_test_dataset==========\n")
        # print(os.path.join(poison_datasets_dir,'test.pt'))
        poisoned_train_dataset = paddle.load(os.path.join(poison_datasets_dir,'train.pdparams'))
        poisoned_test_dataset = paddle.load(os.path.join(poison_datasets_dir,'test.pdparams'))
        
        testset = poisoned_test_dataset
        poisoned_test_indexs = testset.get_poison_indices()
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
       
        # Alreadly exsiting trained model
        print(os.path.join(model_dir, f'backdoor_model.pdparams'))
        model = task_config['model']
        model_state_dict = paddle.load(os.path.join(model_dir, f'backdoor_model.pdparams'))
        result = model.set_state_dict(model_state_dict) 
     
        predict_digits, labels = backdoor.test(model=model, test_dataset=testset)
        benign_acc= compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs])
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs])
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(testset),len(poisoned_test_indexs), len(benign_test_indexs)))                                                                                                                                                
        log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))
    
    







        

       
    
    
 

 
    
  


    
  

 

  
  

