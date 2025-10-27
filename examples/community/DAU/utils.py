import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import argparse
import cv2


parser = argparse.ArgumentParser(description='Select task to execute.')
# choices=['attack',"defense","test",'visualize']
parser.add_argument('--dataset', type=str, default="CIFAR-10", required=False, help='the dataset which is selected to execute,such as ')
parser.add_argument('--subtask', type=str, default="attack", required=False, help='the task which is selected to execute,such as ')
parser.add_argument('--attack', type=str, default="BadNets", required=False, help='the task which is selected to execute,such as ')
parser.add_argument('--method', type=str, default="", required=False, help='the task which is selected to execute,such as ')

class Log:
    log_path = None
    def __init__(self, log_path = None):
        Log.log_path = log_path
    def __call__(self, msg):
        print(msg, end='\n')
        with open(Log.log_path,'a') as f:
            f.write(msg)
    def set_log_path(self, log_path=None):
        Log.log_path = log_path

log = Log(log_path="")

# def load_state(model, state_path):
#     state_dict = torch.load(state_path)
#     # print(model)
#     # print(state_dict.keys())
#     model_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         if k.startswith("module."):
#             k = k.replace("module.", "")
#             model_state_dict[k] = v
#         else:
#             model_state_dict[k] = v
#     model.load_state_dict(model_state_dict, strict=True)
#     return model

# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
    precision = tp / (tp+fp)               # 精确率
    recall = tp / (tp+fn)                  # 召回率
    F1 = (2*tp) / (2*tp+fp+fn)             # F1
    return accuracy, precision, recall, F1
# 计算混淆矩阵
def compute_confusion_matrix(precited,expected):
    predicted = np.array(precited,dtype = int)
    expected = np.array(expected,dtype = int)
    precited = precited.astype(bool)
    expected = expected.astype(bool)

    tp_list = list(precited & expected)    # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
    tn_list = list(~precited & ~expected)    # 将TN的计算结果转换为list
    fn_list = list(~precited & expected)   # 将FN的计算结果转换为list

    tp = tp_list.count(1)                  # 统计TP的个数
    fp = fp_list.count(1)                  # 统计FP的个数
    tn = tn_list.count(1)                  # 统计TN的个数
    fn = fn_list.count(1)                  # 统计FN的个数

    return tp, fp, tn, fn

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = len(target)

    _, pred = output.topk(maxk, 1, True, True)
    # print(pred)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []                    
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate_filter(dataset, predict_poisoned_indices):
    """
    To evaluate the filtering results of poisoned dataset, 
    "dataset" object must support "get_poison_indices()" function
    """

    poison_indices = dataset.get_poison_indices()
    precited = np.zeros(len(dataset))
    precited[predict_poisoned_indices] = 1
    expected = np.zeros(len(dataset))
    expected[poison_indices] = 1
    tp, fp, tn, fn = compute_confusion_matrix(precited,expected)
    # print(f"poison_indices:{poison_indices}, predict_poisoned_indices:{predict_poisoned_indices}\n")
    return tp, fp, tn, fn


def save_img(img, title=None, path=None, **arg):
    '''
        If img is a 3D array,the its shape must be (W,H,C).
    '''
    assert (len(img.shape) == 2 or len(img.shape) == 3), "Image must be 2 or 3 dimensions"
    if len(img.shape) == 3: 
        if img.shape[0] == 1:
            img =  np.squeeze(img,0)
        elif img.shape[0] == 3:
            img =  np.transpose(img, (1, 2, 0))
    plt.figure()
    # print(f"arg:{arg}\n")
    plt.imshow(img,**arg)
    if title is not None:
        plt.title(title)
    plt.axis('off') 
    plt.savefig(path,dpi=600,format='png')

def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:        
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError
    