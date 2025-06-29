# sys
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# mindspore
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import context, nn, Tensor
from mindspore.train import Model
from mindspore.common.initializer import initializer
from mindspore.dataset import vision, transforms
from mindspore.dataset import ImageFolderDataset, Cifar10Dataset
import mindspore.dataset as ds
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net

# numpy
import glob
import numpy as np
import yaml
from typing import Any, Callable, List, Optional, Tuple
import PIL
from PIL import Image
from config.load_config import load_config

config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"task_config.yaml"))
support_tasks = ["ResNet-18_CIFAR-10","ResNet-18_CIFAR-100", "ResNet-18_GTSRB", "ResNet-18_ImageNet","ResNet-18_Tiny-ImageNet","ResNet-18_ImageNetSubset",
                 "VGG_GTSRB",
                 "test_ConvNet"]
support_datasets = ["CIFAR-10","GTSRB"]
support_models = ["ResNet-18","ResNet-34","ResNet-50","ResNet-101","ResNet-152",
                  "VGG",
                  "ConvNet",
                  "Decoupled"]

support_optimizers = ["SGD","Adam","AdamWeightDecay"]
support_losses = ["CrossEntropyLoss","SymmetricCrossEntropyLoss"]

class RCELoss(nn.Cell):
    """
    Reverse Cross Entropy Loss.
    Adapted to MindSpore framework.
    """
    def __init__(self, prob_min=1e-7, one_hot_min=1e-2, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.prob_min = prob_min
        self.one_hot_min = one_hot_min
        self.softmax = nn.Softmax(axis=-1)
        self.log = ms.ops.Log()
        self.sum = ms.ops.ReduceSum()
        self.mean = ms.ops.ReduceMean()

    def construct(self, x, target, weight=None):
        prob = self.softmax(x)
        prob = ms.ops.clip_by_value(prob, clip_value_min=self.prob_min, clip_value_max=1.0)
        one_hot = nn.OneHot(depth=self.num_classes)(target)
        one_hot = ms.ops.clip_by_value(one_hot, clip_value_min=self.one_hot_min, clip_value_max=1.0)
        loss = -1 * self.sum(prob * self.log(one_hot), axis=-1)
        
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == "mean":
            loss = self.mean(loss)
            
        return loss
    
class SCELoss(nn.Cell):
    """Symmetric Cross Entropy adapted to MindSpore."""
    def __init__(self, alpha=0.1, beta=1, num_classes=10, reduction="mean"):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction=reduction)
        self.rce = RCELoss(num_classes=num_classes, reduction=reduction)

    def construct(self, x, target):
        ce_loss = self.ce(x, target)
        rce_loss = self.rce(x, target)
        loss = self.alpha * ce_loss + self.beta * rce_loss
        return loss
    
class VisionGTSRB():
    """GTSRB Dataset adapted for MindSpore"""
    def __init__(self, dataset=None):

        # super(VisionGTSRB, self).__init__()

        self.dataset = dataset
        self.target_size = (32, 32)
        self.data = []
        self.targets = []
        self.classes = []

        for sample in dataset.create_dict_iterator():
            img = sample["image"]
            target = sample["label"]

            if isinstance(img, Image.Image):
                img = img.resize(self.target_size)
                self.data.append(np.array(img))
            else:
                self.data.append(img)
            self.targets.append(target)

        self.classes = [str(label) for label in set(self.targets)]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img) if not isinstance(img, Image.Image) else img
        return img, target
    
    def __len__(self):
        return len(self.data)

def get_dataset(dataset_info=None):
    dataset_type = dataset_info['type']
    assert dataset_type in support_datasets, f"{dataset_type} is not in support_datasets:{support_datasets}"
    datasets_root_dir = dataset_info["dataset_root_dir"]
    datasets_root_dir = os.path.join(BASE_DIR, datasets_root_dir)

    if dataset_type == "CIFAR-10":
        mean = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]  # MindSpore Normalize 不是 [0,1] 范围内的值
        std = [0.2023 * 255, 0.1994 * 255, 0.2010 * 255]
        transform_train = transforms.Compose([
            # vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            # vision.Normalize(mean=mean, std=std)
        ])
        transform_test = transforms.Compose([
            vision.ToTensor(),
            # vision.Normalize(mean=mean, std=std)
        ])
        trainset = Cifar10Dataset(os.path.join(datasets_root_dir,"cifar-10-batches-bin"), usage="train", shuffle=False)
        testset = Cifar10Dataset(os.path.join(datasets_root_dir,"cifar-10-batches-bin"), usage="test", shuffle=False)
        
        trainset = trainset.map(operations=transform_train, input_columns="image")
        testset = testset.map(operations=transform_test, input_columns="image")
        
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = 10
        
        # print(f"len(trainset):{len(trainset)},classes:{classes}\n")
    
    elif dataset_type == "GTSRB":
        transform_train = transforms.Compose([
            vision.Resize((32, 32)),
            vision.ToTensor()
        ])
        transform_test = transforms.Compose([
            vision.Resize((32, 32)),
            vision.ToTensor()
        ])
        
        # Note: MindSpore doesn't have built-in GTSRB dataset, using ImageFolderDataset as alternative
     
        trainset = ImageFolderDataset(os.path.join(datasets_root_dir, "gtsrb/GTSRB/train"))
        testset = ImageFolderDataset(os.path.join(datasets_root_dir, "gtsrb/GTSRB/test"))

        trans = [
            vision.Decode(),                    # 必须先 Decode
            vision.Resize((32, 32)),            # 然后 Resize
            vision.ToTensor()
        ]
        # label_trans = [transforms.TypeCast(mstype.int32)]

        trainset = trainset.map(operations=trans, input_columns="image")
        testset = testset.map(operations=trans, input_columns="image")
        
        # trainset = VisionGTSRB(train_dataset)
        # testset = VisionGTSRB(test_dataset)
        
        classes = None
        num_classes = 43
   
    return trainset, testset, classes, num_classes

def get_model(model_info=None):
    model_type = model_info['type']
    assert model_type in support_models, f"{model_type} is not in support_datasets:{support_models}"
    
    # Note: You'll need to implement these models using MindSpore's nn.Cell
    if model_type == "ResNet-18":
        from models import ResNet
        model = ResNet(18, num_classes=model_info['num_classes'])
    elif model_type == "ConvNet":
        from models import ConvNet
        model = ConvNet(in_channel=model_info["in_channel"], num_classes=model_info['num_classes'])
    elif model_type == "VGG":
        from models import vgg
        model = vgg(model_info['num'], batch_norm=model_info['batch_norm'], num_classes=model_info['num_classes'])
    # Add other model types as needed
    
    return model

def get_loss(loss_info=None):
    loss_type = loss_info['type']
    assert loss_type in support_losses, f"{loss_type} is not in support_datasets:{support_losses}"
    if loss_type == "CrossEntropyLoss":
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    elif loss_type == "SymmetricCrossEntropyLoss":
        loss = SCELoss(alpha=loss_info["alpha"], beta=loss_info["beta"], num_classes=loss_info["num_classes"])
    return loss

def get_optimizer(optimizer=None):
    assert optimizer in support_optimizers, f"{optimizer} is not in support_datasets:{support_optimizers}"
    if optimizer == "SGD":
        from mindspore.nn import SGD
        return SGD
    elif optimizer == "Adam":
        from mindspore.nn import Adam
        return Adam
    elif optimizer == "AdamWeightDecay":
        from mindspore.nn import AdamWeightDecay
        return AdamWeightDecay

def get_scheduler(scheduler):
    if scheduler == "multi_step":
        from mindspore.nn import MultiStepLR
        return MultiStepLR
    elif scheduler == "cosine_annealing":
        from mindspore.nn import CosineDecayLR
        return CosineDecayLR
    else:
        raise ValueError(f"Learning rate scheduler {scheduler} is not supported.")

def get_task_config(task=None):
    assert task in support_tasks, f"{task} is not in support_datasets:{support_tasks}"
    task_config = {
        'train_dataset': None,
        'test_dataset': None,
        'model_type': None,
        'model': None,
        'optimizer': None,
        'lr_scheduler': None,
        "loss": None,
    }
    task_config['train_dataset'], task_config['test_dataset'], _, _ = get_dataset(dataset_info=config[task]["dataset"])
    task_config['model_type'] = config[task]["model"]["type"]
    task_config['model'] = get_model(model_info=config[task]["model"])
    task_config['loss'] = get_loss(loss_info=config[task]["loss"])
    task_config['optimizer'] = get_optimizer(optimizer=config[task]["optimizer"])
    
    if "lr_scheduler" in config[task].keys() and config[task]["lr_scheduler"] is not None:
        task_config['lr_scheduler'] = get_scheduler(config[task]["lr_scheduler"])

    return task_config

def get_task_schedule(task=None):
    assert task in support_tasks, f"{task} is not in support_datasets:{support_tasks}"
    schedule = {}
    for key in config[task]['schedule'].keys():
        schedule[key] = config[task]['schedule'][key]
    return schedule


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    
    task = 'ResNet-18_CIFAR-10'
    task_config = get_task_config(task)
    task_schedule = get_task_schedule(task=task)
    print(config)
    print(task_schedule)
    print(task_schedule['GPU_num'])


    





  






    

