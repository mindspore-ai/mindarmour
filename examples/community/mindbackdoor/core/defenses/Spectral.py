'''
This is the implement of spectral signatures backdoor defense. 
This code is developed based on its official codes, but use the Mindspore instead of TensorFlow. 
(link: https://github.com/MadryLab/backdoor_data_poisoning)

Reference:
[1] Spectral Signatures in Backdoor Attacks. NeurIPS, 2018.
'''

import os
import copy
import time
from typing import Union, Dict, List

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

from .base import Base, _CellWithHook
from ..dataset import DatasetFolder


class Spectral(Base):
    def __init__(
        self,
        model: _CellWithHook,
        loss: nn.Cell,
        poisoned_trainset: DatasetFolder,
        poisoned_testset: DatasetFolder,
        clean_trainset: DatasetFolder,
        clean_testset: DatasetFolder,
        target_label: int,
        percentile: float,
        seed: int = 66
    ):
        super().__init__(seed=seed)

        self.model = model
        self.loss = loss
        self.poisoned_trainset = poisoned_trainset
        self.poisoned_testset = poisoned_testset
        self.clean_trainset = clean_trainset
        self.clean_testset = clean_testset
        self.target_label = target_label
        self.percentile = percentile


    def filter(self, schedule: Union[Dict, None] = None):
        if schedule is not None:
            current_schedule = schedule
        else:
            current_schedule = {
                'device': 'GPU',
                'CUDA_SELECTED_DEVICES': '0',

                'batch_size': 128,
                'num_workers': 1,

                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [150, 180],

                'epochs': 200,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
            }

        if current_schedule.get('device', 'CPU') == 'GPU':
            print('==========Use GPUs to train==========')
            
            selected_devices = current_schedule.get('CUDA_SELECTED_DEVICES', None)
            if selected_devices is None:
                raise AttributeError("CUDA_SELECTED_DEVICES is not set, please set it before training.")
            
            selected_devices = sorted(selected_devices.split(','))
            # we just use the first GPU(the smallest index) to train the model
            selected_device = selected_devices[0]
            
            try:
                ms.set_context(device_target='GPU', device_id=int(selected_device))
            except Exception as e:
                print(f"Error setting GPU context: {e}")
                print("Use CPU to train the autoencoder.")
                ms.set_context(device_target='CPU')
            
        else:
            ms.set_context(device_target='CPU')

        poisoned_label = []

        for i in range(len(self.poisoned_trainset)):
            poisoned_label.append(self.poisoned_trainset[i][1])

        cur_indices = [index for index, label in enumerate(poisoned_label) if label == self.target_label]
        cur_examples = len(cur_indices)

        self.model.set_train()

        full_cov = []
        for index in range(cur_examples):
            cur_image_index = cur_indices[index]
            
            x_batch = np.expand_dims(self.poisoned_trainset[cur_image_index][0], axis=0)
            x_batch = ms.Tensor(x_batch, dtype=ms.float32)
            y_batch = ms.Tensor(self.poisoned_trainset[cur_image_index][0], dtype=ms.int32)

            _ = self.model(x_batch)
            feats = self.model.get_feature('layer4')
            feats = feats.asnumpy().reshape(-1)       # flatten
            full_cov.append(feats)

        total_p = self.percentile

        # full_cov should be a (cur_examples, len(feats)) matrix
        full_cov = np.array(full_cov)
        full_mean = np.mean(full_cov, axis=0, keepdims=True)

        centered_cov = full_cov - full_mean
        _,S,V = np.linalg.svd(centered_cov, full_matrices=False)
        
        print('Top 7 Singular Values: '+ str(S[0:7]))
        eigs = V[0:1]  
        p = total_p
        #shape num_top, num_active_indices
        corrs = np.matmul(eigs, np.transpose(full_cov)) 
        #shape num_active_indices
        scores = np.linalg.norm(corrs, axis=0) 
        # length of score
        print('Length Scores:'+str(len(scores)) )
        p_score = np.percentile(scores, p)
        top_scores = np.where(scores>p_score)[0]
        #print(top_scores)
    
        filtered_poisoned_id = [cur_indices[idx] for idx in top_scores]  # Directly get the indices from cur_indices

        print('removed_inds_length:', len(filtered_poisoned_id))
        print('removed_inds:', filtered_poisoned_id)

        # 2. filtered_benign_id is the remaining indices (poisoned_trainset - filtered_poisoned_id)
        all_indices = set(range(len(self.poisoned_trainset)))
        filtered_poisoned_id_set = set(filtered_poisoned_id)
        filtered_benign_id = sorted(list(all_indices - filtered_poisoned_id_set))

        print('left_inds_length:', len(filtered_benign_id))
        print('left_inds:', filtered_benign_id)

        return filtered_poisoned_id, filtered_benign_id

    def detect(self, poisoned_indices: List[int], schedule: Union[Dict, None] = None):
        """compute metrics: accuracy, precision, recall, F1. 

            Args:
                poisoned_indices (list): poisoned id in clean_dataset
                schedule (dict): schedule for spliting the dataset.            
        """
        filtered_poisoned_id, filtered_benign_id = self.filter(schedule)

        detected = np.zeros(len(self.poisoned_trainset), dtype=int)
        detected[filtered_poisoned_id] = 1

        gt = np.zeros(len(self.poisoned_trainset), dtype=int)
        gt[poisoned_indices] = 1

        TP = np.sum((gt == 1) & (detected == 1))
        FP = np.sum((gt == 0) & (detected == 1))
        TN = np.sum((gt == 0) & (detected == 0))
        FN = np.sum((gt == 1) & (detected == 0))

        accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        return {
            'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN),
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
        }
            