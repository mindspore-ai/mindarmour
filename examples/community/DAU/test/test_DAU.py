# sys 
from copy import deepcopy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
import os.path as osp
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# mindspore 
import mindspore
from mindspore import nn, Tensor
from mindspore.dataset import NumpySlicesDataset

#core
from core.DAU import DAU

from core.attacks import BadNets, Blended, Refool
from config import get_task_config, get_task_schedule, get_defense_config, get_attack_config

#numpy
import numpy as np
import time

#utils
from utils import Log, parser
from utils import compute_indexes, accuracy, compute_accuracy
from utils import evaluate_filter, compute_confusion_matrix
from utils import read_image

# ==================== Set global settings ====================
args = parser.parse_args()
dataset = args.dataset
# ['Attack', 'BackdoorAttack', 'BadNets', 'Blended', 'IAD', 'WaNet', 'SIG', 'Refool']
attack = 'Blended'
defense = 'DAU'

if dataset == "CIFAR-10":
    experiment = 'ResNet-18_CIFAR-10'
    task = 'ResNet-18_CIFAR-10'
    num_classes = 10
elif dataset == "GTSRB": 
    experiment = 'ResNet-18_GTSRB'
    task = 'ResNet-18_GTSRB'
    num_classes = 43

experiment_dir = os.path.join(BASE_DIR, "save/")
y_target = 0
poisoned_rate = 0.1
backdoor_model_dir = os.path.join(experiment_dir, f'{task}/attack/{attack}/model/y_target_{y_target}_poison_{poisoned_rate}/')
backdoor_model_file = f"backdoor_model.ckpt"

# dataset
poison_datasets_dir = os.path.join(experiment_dir,f'{task}/attack/{attack}/datasets/poisoned_data/y_target_{y_target}_poison_{poisoned_rate}/')
poison_train_datasets_file = 'poisoned_train_dataset.npz'
poison_test_datasets_file = 'poisoned_test_dataset.npz'


work_dir = os.path.join(experiment_dir, f'{task}/{defense}_for_{attack}/y_target_{y_target}_poison_{poisoned_rate}/')

# ====================Load global config ====================
task_config = get_task_config(task = task)
schedule = get_task_schedule(task = task)
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir

defense_schedule = get_defense_config(defense_strategy = defense)
defense_schedule['anti_learning']['num_classes'] = num_classes
defense_schedule['filter']['num_classes'] = num_classes
defense_schedule['schedule'] = schedule
defense_schedule['work_dir'] = work_dir

# Directory setup
dirs = []
dir = os.path.join(work_dir, f"anti_learning/")
anti_learning_model_dir = os.path.join(dir, f"model/")
anti_learning_data_dir = os.path.join(dir, f"data/")
anti_learning_filter_dir = os.path.join(anti_learning_data_dir, "data_filter/")
anti_dirs = [anti_learning_model_dir, anti_learning_data_dir, anti_learning_filter_dir]
dirs.extend(anti_dirs)

dir = os.path.join(work_dir, f"unlearning/")
unlearning_model_dir = os.path.join(dir, f"model/")
unlearning_dirs = [unlearning_model_dir] 
dirs.extend(unlearning_dirs)

for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# anti_learning
alpha = defense_schedule["anti_learning"]["alpha"]
beta = defense_schedule["anti_learning"]["beta"]  
anti_learning_model_file = f"anti_learning_model.ckpt"
anti_learning_filter_file = f"anti_learning_filter_alpha_{alpha}_beta_{beta}.npz"

# unlearning
unlearning_model_file =  f"unlearning_model.ckpt"

log = Log(osp.join(work_dir, 'log.txt'))
t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
log(msg)

dau = DAU(
    task_config,
    defense_schedule
)

def get_attack_object(attack="BadNets", dataset="CIFAR-10"):
    task_config = None
    attack_schedule = get_attack_config(attack_strategy=attack, dataset=dataset)
    global save_dataset, load_dataset
    if attack == "BadNets":
        attack_object = BadNets(
            task_config,
            attack_schedule
        )
        from core.attacks.BadNets import save_dataset as badnets_save
        from core.attacks.BadNets import load_dataset as badnets_load
        save_dataset, load_dataset = badnets_save, badnets_load

    elif attack == "Blended":
        attack_object = Blended(
            task_config,
            attack_schedule
        )
        from core.attacks.Blended import save_dataset as blended_save
        from core.attacks.Blended import load_dataset as blended_load
        save_dataset, load_dataset = blended_save, blended_load
  
    elif attack == "Refool":
        reflection_images = []
        reflection_data_dir = os.path.join(BASE_DIR,"datasets/VOCdevkit/VOC2012/JPEGImages/")
        reflection_image_path = os.listdir(reflection_data_dir)
        reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
        attack_schedule['reflection_candidates'] = reflection_images
        attack_object = Refool(
            task_config,
            attack_schedule
        )
        from core.attacks.Refool import save_dataset as refool_save
        from core.attacks.Refool import load_dataset as refool_load
        save_dataset, load_dataset = refool_save, refool_load

    return attack_object

def set_global_settings(dataset="CIFAR-10", attack="BadNets"):
    global experiment, task, num_classes, experiment_dir, y_target, poisoned_rate
    global work_dir, backdoor_model_dir, backdoor_model_file, poison_datasets_dir, poison_train_datasets_file, poison_test_datasets_file
    global task_config, schedule, defense_schedule
    global anti_learning_model_dir, anti_learning_data_dir, anti_learning_filter_dir, unlearning_model_dir, anti_learning_model_file, anti_learning_filter_file, unlearning_model_file
    global dau
    global result_dict

    result_dict = {
        
        # 以下部分是本场景没用到的
        "status": "running",
        "epoches": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "text_data": "",
        "asr": 1.23,
        "acc": 0.89,
        "trust_score": [],

        # 以下部分是本场景用到的输出数据 
        "loss": [],
        "asr": [],
        "acc": [],
        "bar_vals": [],
        "bar_names": ["真阳率", "假阳率"]
    }

    if dataset == "CIFAR-10":
        experiment = 'ResNet-18_CIFAR-10'
        task = 'ResNet-18_CIFAR-10'
        num_classes = 10
    elif dataset == "GTSRB": 
        experiment = 'ResNet-18_GTSRB'
        task = 'ResNet-18_GTSRB'
        num_classes = 43

    experiment_dir = os.path.join(BASE_DIR, "save/")
    y_target = 0
    poisoned_rate = 0.1
    work_dir = os.path.join(experiment_dir, f'{task}/{defense}_for_{attack}/y_target_{y_target}_poison_{poisoned_rate}/')
    backdoor_model_dir = os.path.join(experiment_dir, f'{task}/attack/{attack}/model/y_target_{y_target}_poison_{poisoned_rate}/')
    backdoor_model_file = f"backdoor_model.ckpt"
    
    poison_datasets_dir = os.path.join(experiment_dir,f'{task}/attack/{attack}/datasets/poisoned_data/y_target_{y_target}_poison_{poisoned_rate}/')
    poison_train_datasets_file = 'poisoned_train_dataset.npz'
    poison_test_datasets_file = 'poisoned_test_dataset.npz'

    # Load configs
    task_config = get_task_config(task = task)
    schedule = get_task_schedule(task = task)
    schedule['experiment'] = experiment
    schedule['work_dir'] = work_dir

    defense_schedule = get_defense_config(defense_strategy = defense)
    defense_schedule['anti_learning']['num_classes'] = num_classes
    defense_schedule['filter']['num_classes'] = num_classes
    defense_schedule['schedule'] = schedule
    defense_schedule['work_dir'] = work_dir

    # Create directories
    dirs = []
    dir = os.path.join(work_dir, f"anti_learning/")
    anti_learning_model_dir = os.path.join(dir, f"model/")
    anti_learning_data_dir = os.path.join(dir, f"data/")
    anti_learning_filter_dir = os.path.join(anti_learning_data_dir, "data_filter/")
    anti_dirs = [anti_learning_model_dir, anti_learning_data_dir, anti_learning_filter_dir]
    dirs.extend(anti_dirs)

    dir = os.path.join(work_dir, f"unlearning/")
    unlearning_model_dir = os.path.join(dir, f"model/")
    unlearning_dirs = [unlearning_model_dir] 
    dirs.extend(unlearning_dirs)

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Set file names
    alpha = defense_schedule["anti_learning"]["alpha"]
    beta = defense_schedule["anti_learning"]["beta"]  
    anti_learning_model_file = f"anti_learning_model.ckpt"
    anti_learning_filter_file = f"anti_learning_filter_alpha_{alpha}_beta_{beta}.npz"
    unlearning_model_file =  f"unlearning_model.ckpt"

    dau = DAU(
        task_config,
        defense_schedule
    )
    
def unlearning(args):
    dataset = args["data_name"]
    attack = args['custom_parameter']['attack'] 
    set_global_settings(dataset=dataset, attack=attack)

    # load model and dataset
    model_dir, model_file = backdoor_model_dir, backdoor_model_file
    model = task_config['model']
    
    # Load model parameters
    param_dict = mindspore.load_checkpoint(os.path.join(model_dir, model_file))
    mindspore.load_param_into_net(model, param_dict)

    # create_poisoned_dataset
    attack_object = get_attack_object(attack = attack)
    clean_train_dataset = task_config['train_dataset']
    clean_test_dataset = task_config['test_dataset']

    poisoned_train_dataset = attack_object.create_poisoned_dataset(deepcopy(clean_train_dataset), y_target=y_target, poisoned_rate=poisoned_rate, train=False)
    poisoned_test_dataset = attack_object.create_poisoned_dataset(deepcopy(clean_test_dataset), y_target=y_target, poisoned_rate=poisoned_rate, train=True)
    
    # get initial poisoned data pool
    poison_indices = poisoned_train_dataset.get_poison_indices()
    pre_filter_rate = 0.01
    random_poison_indices = np.random.choice(poison_indices, size=int(len(poisoned_train_dataset) * pre_filter_rate), replace = False)
    other_indices = np.setdiff1d(np.arange(len(poisoned_train_dataset)),random_poison_indices)
    init_poison_data_pool, init_clean_data_pool = random_poison_indices, other_indices

    # anti_learning
    defense_schedule["anti_learning"]["pre_train_model"] = model
    defense_schedule["anti_learning"]["init_poison_data_pool"] = init_poison_data_pool
    defense_schedule["anti_learning"]["init_clean_data_pool"] = init_clean_data_pool
    poison_data_pool, clean_data_pool = dau.anti_learning(pre_train_model=deepcopy(model), dataset=poisoned_train_dataset, schedule=defense_schedule)

    poison_indices = poisoned_train_dataset.get_poison_indices()
    expected = np.zeros(len(poisoned_train_dataset))
    expected[poison_indices] = 1

    precited = np.zeros(len(poisoned_train_dataset))
    precited[poison_data_pool] = 1
    
    tp, fp, tn, fn = compute_confusion_matrix(precited,expected)
    tpr = tp / (tp + fn) * 100
    fpr = fp / (fp + tn) * 100
    result_dict["bar_vals"] = [tpr, fpr]

    defense_schedule["unlearning"]["clean_data_pool"] = clean_data_pool 
    defense_schedule["unlearning"]["poison_data_pool"] = poison_data_pool 

    # unlearning 
    dau.unlearning(args=args, model=deepcopy(model), dataset=poisoned_train_dataset, test_dataset=poisoned_test_dataset, schedule=defense_schedule, result_dict=result_dict)
    model = dau.get_model()

    mindspore.save_checkpoint(model, os.path.join(unlearning_model_dir, unlearning_model_file))
    log("Save model to" + os.path.join(unlearning_model_dir, unlearning_model_file))


if __name__ == "__main__":

    """
    # anti_learning

    python test_DAU.py --subtask "anti_learning" --dataset "CIFAR-10"
    python test_DAU.py --subtask "anti_learning" --dataset "GTSRB"
    python test_DAU.py --subtask "anti_learning" --dataset "Tiny-ImageNet"

    # unlearning

    python test_DAU.py --subtask "unlearning"  --dataset "CIFAR-10" 
    python test_DAU.py --subtask "unlearning"  --dataset "GTSRB" 
    python test_DAU.py --subtask "unlearning"  --dataset "Tiny-ImageNet" 

    """
      
    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    dau = DAU(
        task_config,
        defense_schedule
    )

    if args.subtask == "anti_learning": 
         
        # Load poisoned datasets
        attack_object = get_attack_object(attack = attack)

        filter_res = {"poison_data_pool":None, "clean_data_pool":None}   
        poisoned_train_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_train_datasets_file))
        poisoned_test_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_test_datasets_file))

        log("=======================test backdoor model====================\n")

        model = task_config['model']
        param_dict = mindspore.load_checkpoint(os.path.join(backdoor_model_dir, backdoor_model_file))
        mindspore.load_param_into_net(model, param_dict)

        testset = poisoned_test_dataset
        poisoned_test_indices = testset.get_poison_indices()
        benign_test_indices = list(set(range(len(testset))) - set(poisoned_test_indices))

        predict_digits, labels = dau.test(model=model, test_dataset=testset)
        poisoned_test_indices = poisoned_test_indices.tolist()

        benign_acc = compute_accuracy(predict_digits[benign_test_indices], labels[benign_test_indices])
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indices], labels[poisoned_test_indices])
        
        log(f"Total samples:{len(testset)}, poisoning samples:{len(poisoned_test_indices)}, benign samples:{len(benign_test_indices)}\n")                                                                                                                                                
        log(f"Benign_accuracy:{benign_acc[0]}, poisoning_accuracy:{poisoned_acc[0]}\n")

        log("=======================get initial poisoned data pool====================\n")
    
        poison_indices = poisoned_train_dataset.get_poison_indices()
        pre_filter_rate = 0.01
        random_poison_indices = np.random.choice(poison_indices, size=int(len(poisoned_train_dataset) * pre_filter_rate), replace = False)
        other_indices = np.setdiff1d(np.arange(len(poisoned_train_dataset)),random_poison_indices)
        init_poison_data_pool, init_clean_data_pool = random_poison_indices, other_indices

        tp, fp, tn, fn = evaluate_filter(poisoned_train_dataset, init_poison_data_pool) 
        log(f"init_poison_data_pool:{len(init_poison_data_pool)}, tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}\n")
        
        log("=======================anti_learning====================\n")

        defense_schedule["anti_learning"]["pre_train_model"] = model
        defense_schedule["anti_learning"]["init_poison_data_pool"] = init_poison_data_pool
        defense_schedule["anti_learning"]["init_clean_data_pool"] = init_clean_data_pool

        poison_data_pool, clean_data_pool = dau.anti_learning(pre_train_model=model, dataset=poisoned_train_dataset, schedule=defense_schedule)
        
        filter_res["poison_data_pool"], filter_res["clean_data_pool"] = poison_data_pool, clean_data_pool
        
        filter_dir, filter_file = anti_learning_filter_dir, anti_learning_filter_file
        np.savez(os.path.join(filter_dir, filter_file), **filter_res)
        log(f"Save the result of filtering into {os.path.join(filter_dir, filter_file)}\n")
            
        model = dau.get_model()
        mindspore.save_checkpoint(model, os.path.join(anti_learning_model_dir, anti_learning_model_file))
        log("Save model to" + os.path.join(anti_learning_model_dir, anti_learning_model_file))

    elif args.subtask == "unlearning":
        # Load poisoned datasets
        attack_object = get_attack_object(attack = attack)

        # Get model
        model = task_config['model'] 
        model_dir, model_file = backdoor_model_dir, backdoor_model_file
        
        # Load model parameters
        param_dict = mindspore.load_checkpoint(os.path.join(model_dir, model_file))
        mindspore.load_param_into_net(model, param_dict)

        # Get poisoned dataset
        poisoned_train_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_train_datasets_file))
        poisoned_test_dataset = load_dataset(os.path.join(poison_datasets_dir, poison_test_datasets_file))

        poison_data_pool = poisoned_train_dataset.get_poison_indices()
        clean_data_pool = np.setdiff1d(np.arange(len(poisoned_train_dataset)),poison_data_pool)
        defense_schedule["unlearning"]["clean_data_pool"] = clean_data_pool 
        defense_schedule["unlearning"]["poison_data_pool"] = poison_data_pool 
        log(f"clean_data_pool:{len(clean_data_pool)}, poison_data_pool:{len(poison_data_pool)}\n")
        
        tp, fp, tn, fn = evaluate_filter(poisoned_train_dataset, poison_data_pool)
        log(f"tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}\n")
        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        log(f"accuracy:{accuracy}, precision:{precision}, recall:{recall}, F1:{F1}\n")  
        
        log("=======================test backdoor model====================\n")
        testset = poisoned_test_dataset
        poisoned_test_indexs = testset.get_poison_indices()
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
        poisoned_test_indexs = poisoned_test_indexs.tolist()
       
        predict_digits, labels = dau.test(model=model, test_dataset=testset)
        benign_acc = compute_accuracy(predict_digits[benign_test_indexs], labels[benign_test_indexs])
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs], labels[poisoned_test_indexs])
        log(f"Total samples:{len(testset)}, poisoning samples:{len(poisoned_test_indexs)}, benign samples:{len(benign_test_indexs)}\n")                                                                                                                                                
        log(f"Benign_accuracy:{benign_acc[0]}, poisoning_accuracy:{poisoned_acc[0]}\n")

        log("=======================unlearning====================\n")
        dau.unlearning(model=model, dataset=poisoned_train_dataset, test_dataset=poisoned_test_dataset, schedule=defense_schedule)
        model = dau.get_model()
        unlearning_model_file = f"unlearning_model_adv_loss_{defense_schedule['unlearning']['unlearning_loss']}.ckpt"
       
        mindspore.save_checkpoint(model, os.path.join(unlearning_model_dir, unlearning_model_file))
        log("Save model to" + os.path.join(unlearning_model_dir, unlearning_model_file))
        
        # test model after unleaning
        log("Test model after unlearning\n")
        test_dataset = poisoned_test_dataset
        poisoned_test_indexs = test_dataset.get_poison_indices()
        benign_test_indexs = list(set(range(len(test_dataset))) - set(poisoned_test_indexs))
        predict_digits, labels = dau.test(model=model, test_dataset=test_dataset.create_tuple_iterator(output_numpy=True))
        benign_acc = compute_accuracy(predict_digits[benign_test_indexs], labels[benign_test_indexs])
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs], labels[poisoned_test_indexs])
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(test_dataset),len(poisoned_test_indexs),len(benign_test_indexs)))                                                                                                                                                
        log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))