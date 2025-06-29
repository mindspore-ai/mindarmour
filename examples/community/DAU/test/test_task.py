# sys 
import os
import os.path as osp
os.environ['DEVICE_ID'] = "0,1,2,3,4,5"  # Changed from CUDA_VISIBLE_DEVICES to DEVICE_ID for MindSpore
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(f"BASE_DIR:{BASE_DIR}\n")

# mindspore
import mindspore as ms
from mindspore import context, nn, Tensor
from mindspore.train import Model
from mindspore.common.initializer import initializer
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net

#core
from core.Base import Base
from config import get_task_config, get_task_schedule
#numpy
import numpy as np
import time

#utils
from utils import Log, parser, compute_accuracy

"""
First, test running a normal task: Take ResNet-18_CIFAR-10 task as an example to run through the entire logic.
"""

# ==================== Set global settings ====================
args = parser.parse_args()
dataset = args.dataset
# "No_attack" "BadNets", "Blended", "WaNet", "Refool", "IAD", "LabelConsistent"

attack = "No_attack"

if dataset == "CIFAR-10":
    experiment = 'ResNet-18_CIFAR-10'
    task = 'ResNet-18_CIFAR-10'
    num_classes = 10

elif dataset == "GTSRB": 
    experiment = 'ResNet-18_GTSRB'
    task = 'ResNet-18_GTSRB'
    num_classes = 43
    
elif dataset == "Tiny-ImageNet":
    experiment = f'ResNet-18_Tiny-ImageNet' 
    task = 'ResNet-18_Tiny-ImageNet'
    num_classes = 200

experiment_dir = os.path.join(BASE_DIR, "save/")
work_dir = os.path.join(experiment_dir, f'{task}/{attack}')

# ====================Load global config ====================
task_config = get_task_config(task=task)

schedule = get_task_schedule(task=task)
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir

# # Set MindSpore context
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")  # Changed from CUDA to GPU for MindSpore

# 1. model_dir 
dirs = []
model_dir = os.path.join(work_dir, f'{task}/model/')
model_file = f"model.ckpt"  # Changed from .pdparams to .ckpt for MindSpore
dirs.extend([model_dir])

for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

log = Log(osp.join(work_dir, 'log.txt'))
t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
log(msg)


def set_global_settings(dataset="CIFAR-10", attack="BadNets"):
    global experiment, task, num_classes, experiment_dir
    global work_dir, model_dir, model_file
    global task_config, schedule
    global base_object
 
    if dataset == "CIFAR-10":
        experiment = 'ResNet-18_CIFAR-10'
        task = 'ResNet-18_CIFAR-10'
        num_classes = 10
    elif dataset == "CIFAR-100":
        experiment = 'ResNet-18_CIFAR-100'
        task = 'ResNet-18_CIFAR-100'
        num_classes = 100
    elif dataset == "GTSRB": 
        experiment = 'ResNet-18_GTSRB'
        task = 'ResNet-18_GTSRB'
        num_classes = 43
    elif dataset == "ImageNet":
        task = 'ResNet-18_ImageNet'
        num_classes = 10
    elif dataset == "Tiny-ImageNet":
        experiment = f'ResNet-18_Tiny-ImageNet' 
        task = 'ResNet-18_Tiny-ImageNet'
        num_classes = 200

    experiment_dir = os.path.join(BASE_DIR, "save/")
    work_dir = os.path.join(experiment_dir, f'{task}/{attack}')

    # Load global config
    task_config = get_task_config(task=task)
    schedule = get_task_schedule(task=task)
    schedule['experiment'] = experiment
    schedule['work_dir'] = work_dir

    # model_dir 
    dirs = []
    model_dir = os.path.join(work_dir, f'{task}/model/')
    model_file = f"model.ckpt"  # Changed for MindSpore
    dirs.extend([model_dir])

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)

    base_object = Base(task=task_config, schedule=schedule) 


def run(args):
    dataset = args["data_name"]
    attack = args['custom_parameter']['attack'] 
    set_global_settings(dataset=dataset, attack=attack)

    #Train and get backdoor model
    log("\n==========Train on clean dataset and get clean model==========\n")
    # Show the structure of the model
    log(str(task_config['model']))
    base_object.train(train_dataset=task_config["train_dataset"], test_dataset=task_config["test_dataset"], schedule=schedule)
    clean_model = base_object.get_model()
    save_checkpoint(clean_model, os.path.join(model_dir, model_file))  # Changed from paddle.save to save_checkpoint
    log("Save clean model to" + os.path.join(model_dir, model_file))


if __name__ == "__main__":
    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)

    base_object = Base(task=task_config, schedule=schedule) 
   
    """
    # train
    python test_task.py --subtask "train" --dataset "CIFAR-10"
    python test_task.py --subtask "train" --dataset "GTSRB"
    python test_task.py --subtask "train" --dataset "Tiny-ImageNet"

    # test
    python test_task.py --subtask "test" --dataset "CIFAR-10"
    python test_task.py --subtask "test" --dataset "GTSRB"
    python test_task.py --subtask "test" --dataset "Tiny-ImageNet"

    """
    if args.subtask == "train":  
        log("\n==========Train on clean dataset and get clean model==========\n")
        # Show the structure of the model
        # log(str(task_config['model']))
        base_object.train(train_dataset=task_config["train_dataset"], test_dataset=task_config["test_dataset"], schedule=schedule)
        clean_model = base_object.get_model()

        save_checkpoint(clean_model, os.path.join(model_dir, model_file))  # Changed for MindSpore
        log("Save clean model to" + os.path.join(model_dir, model_file))

    elif args.subtask == "test": 

        train_dataset = task_config["train_dataset"]
        test_dataset = task_config["test_dataset"]
        model = task_config["model"]
      
        param_dict = load_checkpoint(os.path.join(model_dir, model_file))
        param_not_load, _ = load_param_into_net(model, param_dict)
        # print(param_not_load)

        predict_digits, labels = base_object.test(model=model, test_dataset=test_dataset)


