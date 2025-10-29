"""
This is the implementation of Beatrix[1].

[1] The "Beatrix" Resurrections: Robust Backdoor Detection via Gram Matrices. (NDSS 2023)
"""

from typing import Literal, List, Union, Dict

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
from sklearn import metrics

from .base import Base, _CellWithHook
from ..dataset import DatasetFolder

class Beatrix(Base):
    def __init__(
        self,
        model: _CellWithHook,
        order_list: List[int] = list(np.arange(1, 9)),
        seed: int = 66,
    ):
        super().__init__(seed=seed)

        self.model = model
        self.model.set_train(False)

        self.order_list = order_list

    @classmethod
    def Gram_p_matrix(cls, x: np.ndarray, p: int):
        """
        Calculate the Gram matrix of the feature map
        
        Args:
            x: the feature map
            p: the order of the Gram matrix
            
        Returns:
            The Gram matrix
        """

        # Expand the dimension of the feature map
        if x.ndim == 3:
            x = np.expand_dims(x, 0)  # (1, C, H, W)
        B, C = x.shape[:2]
        out = x ** p
        out = out.reshape(B, C, -1)  # (B, C, N)
        gram = np.matmul(out, out.transpose(0, 2, 1))  # (B, C, C)
        # upper triangle
        gram = np.triu(gram)
        # p-th root
        gram = np.sign(gram) * (np.abs(gram) ** (1 / p))
        gram = gram.reshape(B, -1)
        return gram

    def get_deviations(self, features, medians, mads, order_list):
        """
        Calculate the deviation score of the feature vector 
        
        Args:
            features: list, each element shape=(B, C, H, W)
            medians: list, each element is the median of gram_p, shape=(1, -1) or (1, num_elements)
            mads: list, each element is the absolute median deviation of gram_p, shape=(1, -1)
            order_list: list, the order list of the Gram matrix
        Returns:
            deviations: shape=(sum_batch, 1) the deviation score
        """
        deviations = []
        for feat in features:
            dev = 0
            for p, P in enumerate(order_list):
                g_p = self.Gram_p_matrix(feat, P)  # (B, -1)
                # numpy broadcast, sum by row, keep axis=1
                dev += np.sum(np.abs(g_p - medians[p]) / (mads[p] + 1e-6), axis=1, keepdims=True)
            deviations.append(dev)
        deviations = np.concatenate(deviations, axis=0)
        return deviations

    def _detect(self, dataset: DatasetFolder, schedule: Union[Dict, None] = None):
        if schedule is not None:
            current_schedule = schedule
        else:
            current_schedule = {
                'device': 'GPU',
                'CUDA_SELECTED_DEVICES': '0',

                'batch_size': 128,
                'num_workers': 1,
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

        data_loader = dataset.to_generator_dataset(
            batch_size=current_schedule['batch_size'],
            shuffle=True,
            num_parallel_workers=current_schedule['num_workers']
        )

        self.model.set_train(False)
        
        features = []
        pred_correct_mask = []

        for _, (batch_x, batch_y) in enumerate(data_loader):

            logits = self.model(batch_x)
            feats = self.model.get_feature('layer4')

            original_preds = np.argmax(logits.asnumpy(), axis=1)
            mask = np.equal(original_preds, batch_y.asnumpy())

            pred_correct_mask.append(mask)
            features.append(feats)

        features = np.concatenate(features, axis=0)
        pred_correct_mask = np.concatenate(pred_correct_mask, axis=0)

        return features[pred_correct_mask], pred_correct_mask

    def detect(self, clean_dataset: DatasetFolder, poisoned_dataset: DatasetFolder, schedule: Union[Dict, None] = None):
        """
        Detect the backdoor in the dataset, and return TP, FP, TN, FN, accuracy, precision, recall, f1
        
        Args:
            clean_dataset: the clean dataset
            poisoned_dataset: the poisoned dataset
            schedule: the schedule for the detection
        """
        clean_features, clean_mask = self._detect(clean_dataset, schedule)
        poisoned_features, poisoned_mask = self._detect(poisoned_dataset, schedule)

        medians = []
        mads = []
        for P in self.order_list:
            gram_p = self.Gram_p_matrix(clean_features, P)
            median = np.median(gram_p, axis=0, keepdims=True)[0]
            mad = np.median(np.abs(gram_p - median), axis=0, keepdims=True)[0]
            medians.append(median)
            mads.append(mad)

        clean_deviations = self.get_deviations(clean_features, medians, mads, self.order_list)
        poisoned_deviations = self.get_deviations(poisoned_features, medians, mads, self.order_list)

        # concatenate the detection target/score
        y_true = np.concatenate([
            np.zeros_like(clean_deviations),
            np.ones_like(poisoned_deviations)
        ])
        y_score = np.concatenate([
            clean_deviations,
            poisoned_deviations
        ]).flatten()

        # threshold setting: the paper suggests mean+2std (you can also try median+2std)
        threshold = np.median(clean_deviations) + 2 * np.std(clean_deviations)
        y_pred = (y_score >= threshold).astype(int)

        # calculate metrics
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        f1 = metrics.f1_score(y_true, y_pred)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)

        print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
        print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
        print("AUC: {:.4f}".format(auc))
        print(f"F1 Score: {f1:.4f}")

        return {
            'clean_deviations': clean_deviations,
            'poisoned_deviations': poisoned_deviations,
            'threshold': threshold,
            'auc': auc,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion': dict(tp=tp, fp=fp, tn=tn, fn=fn),
        }