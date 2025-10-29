'''
This is the implement of TUAP [1].

[1] Clean-Label Backdoor Attacks on Video Recognition Models. CVPR, 2020.
'''

import copy
import random


from PIL import Image
from typing import Optional, Union, Tuple, List, Dict, Any
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops

from .base import Base, ModifyTarget
from ..dataset import DatasetFolder, Compose

class AddTrigger:
    def __init__(self, pattern: np.ndarray, mask: np.ndarray):
        # the range of pattern lies in [-1, 1]:
        # format of pattern is (C, H, W) cuz the pattern would be optmized in the training process
        self.pattern = pattern
        self.mask = mask
        self.res = self.mask * self.pattern
        
    def add_trigger(self, img: np.ndarray):
        # may change later
        return (self.res + img).astype(np.float32)
    
    def __call__(self, img: Optional[Union[np.ndarray, Image.Image]]) -> Optional[Union[np.ndarray, Image.Image]]:
        if type(img) == Image.Image:
            # still in (H, W, C)
            # transform to (C, H, W)
            img: np.ndarray = np.array(img)  # Convert to numpy (H, W, C) or (H, W)
    
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)  # (1, H, W)
            elif img.ndim == 3 and img.shape[2] == 3:
                img = np.transpose(img, (2, 0, 1))  # (3, H, W)
            else:
                raise ValueError("Unsupported input image shape.")
            
            # Add trigger
            img = self.add_trigger(img)

            # Convert to uint8 before Image.fromarray
            # but the image of poisoned_img is [0, 255]
            # img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img = img.astype(np.uint8)
            
            if img.shape[0] == 1:
                img = Image.fromarray(img.squeeze(0), mode='L')
            elif img.shape[0] == 3:
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            else:
                raise ValueError("Invalid image size after trigger.")
            
            return img
        
        elif type(img) == np.ndarray:
            # tensor format default in [-1, 1] in (C, H, W)
            img = self.add_trigger(img)
            return img
            # done
            
        else:
            raise TypeError("Unsupported image type.")
    

class PoisonedDatasetFolder(DatasetFolder):
    def __init__(
        self,
        benign_dataset: DatasetFolder,
        y_target: int,
        is_train_set: bool,
        poisoned_rate: float,
        pattern: np.ndarray,
        mask: np.ndarray,
        poisoned_transform_index: int,
        poisoned_target_transform_index: int,
    ):
        super(PoisonedDatasetFolder, self).__init__(
            root=benign_dataset.root,
            transform=benign_dataset.transform,
            target_transform=benign_dataset.target_transform,
            loader=benign_dataset.loader,
            extensions=benign_dataset.extensions,
        )

        self.is_train_set = is_train_set
        self.benign_dataset = benign_dataset
        self.y_target_ori = int(y_target)
        self.poisoned_rate = poisoned_rate
        
        if self.is_train_set: # for training
            self.poisoned_set = self.gen_poisoned_index()
        else:
            total_num = len(self.benign_dataset)
            poisoned_num = int(total_num * poisoned_rate)
            tmp_list = list(range(total_num))
            random.shuffle(tmp_list)
            self.poisoned_set = copy.deepcopy(tmp_list[:poisoned_num])
            
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddTrigger(pattern, mask))
        
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
            
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
         
            
    def gen_poisoned_index(self):
        target_label_list = []
        for (index, t) in enumerate(self.benign_dataset.targets):
            if t == self.y_target_ori:
                target_label_list.append(index)

        num_target_sample = len(target_label_list)
        np.random.shuffle(np.array(target_label_list))

        poisoned_num = int(num_target_sample * self.poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        poisoned_set = frozenset(target_label_list[:poisoned_num])
        return poisoned_set
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
         
        target = np.int32(target)
            
        return sample, target
    

class UAP:
    def __init__(
        self,
        model: nn.Cell,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        target_class: int,
        mask=None,
        p_samples: float = 0.01,
    ):
        self.model = model
        self.mask = mask
        self.target_class = target_class
        self.trainset = train_dataset
        self.testset = test_dataset
        self.p_samples = p_samples
        
        self.num_samples = int(self.p_samples * len(self.trainset)) + 1
        
        self.test_loader = test_dataset.to_generator_dataset(
            batch_size=200,
            shuffle=False,
            num_parallel_workers=1
        )
        
    def deepfool_target(
        self, 
        image: ms.Tensor,
        num_classes: int,
        overshoot: float = 0.02,
        max_iter: int = 50,
        ):
        """
           :param image: Image of size (C, H, W)
           :param num_classes: number of classes (limits the number of classes to test against, by default = 10)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :param max_iter: maximum number of iterations for deepfool (default = 50)
           :return: minimal perturbation that fools the classifier, number of iterations that it required,
           new estimated_label and perturbed image
        """
        input_shape = image.asnumpy().shape
        image = image.expand_dims(0)
        
        f_image = self.model(ms.Parameter(image.copy(), requires_grad=True))
        f_image = f_image.asnumpy().flatten()
        
        I = f_image.argsort()[::-1]
        I = I[:num_classes]
        
        # clean_label = I[0]
        clean_label = int(I[0])
        
        pert_image = copy.deepcopy(image)
        
        r_tot = np.zeros(input_shape)
        
        loop_i = 0
        
        x = ms.Parameter(pert_image, requires_grad=True)
        k_i = clean_label
        
        while True:
            # with ms.context.set_grad(True):
            x = ms.Parameter(pert_image.copy(), requires_grad=True)
            logits = self.model(x)
            logits = logits[0]

            def loss_target_fn(x):
                logits = self.model(x)
                return -logits[0, self.target_class]
            def loss_clean_fn(x):
                logits = self.model(x)
                return -logits[0, clean_label]

            grad_target_fn = ms.grad(loss_target_fn)
            grad_clean_fn = ms.grad(loss_clean_fn)

            grad_target = grad_target_fn(pert_image)
            grad_clean = grad_clean_fn(pert_image)
            
            # # 目标类别和原始类别的loss
            # loss_target = -logits[self.target_class]
            # loss_clean = -logits[clean_label]

            # grad_fn = ms.grad(loss_target, x)
            # grad_target = grad_fn(x)
            # grad_fn_clean = ms.grad(loss_clean, x)
            # grad_clean = grad_fn_clean(x)

            w_k = (grad_target - grad_clean)
            if self.mask is not None:
                w_k = w_k * self.mask
            f_k = (logits[self.target_class] - logits[clean_label]).asnumpy()

            w_k_np = w_k.asnumpy()
            pert_k = abs(f_k) / (np.linalg.norm(w_k_np.flatten()) + 1e-8)

            r_i = (pert_k + 1e-4) * w_k_np / (np.linalg.norm(w_k_np) + 1e-8)
            # r_tot += r_i
            r_tot = np.float32(r_tot + r_i)
            pert_image = ms.Tensor(image.asnumpy() + (1 + overshoot) * r_tot, ms.float32)

            # with ms.context.set_grad(False):
            logits_new = self.model(pert_image)
            k_i = np.argmax(logits_new.asnumpy().flatten())
            loop_i += 1
            if k_i == self.target_class or loop_i >= max_iter:
                break

        return (1 + overshoot) * r_tot, loop_i, k_i, pert_image
        
    
    def proj_lp(self, perturbation, epsilon, p_norm):
        """
            Project on the lp ball centered at 0 and of radius epsilon, SUPPORTS only p = 2 and p = Inf for now
            :param perturbation: Perturbation of size CxHxW
            :param epsilon: Controls the l_p magnitude of the perturbation (default = 10/255.0)
            :param p_norm: Norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
            :return:
        """
        if p_norm == 2:
            norm = ops.Norm()(perturbation.view(-1), 2)
            factor = min(1, epsilon / norm.asnumpy())
            perturbation = perturbation * factor
        elif p_norm == np.inf:
            perturbation = ops.clip_by_value(perturbation, -epsilon, epsilon)
        else:
            raise ValueError('Only p=2 or p=inf is supported.')
        return perturbation
             
    
    def universal_perturbation(
        self,
        delta=0.2, 
        max_iter_uni=40, 
        epsilon=10.0/255,
        p_norm=np.inf, 
        num_classes=10, 
        overshoot=0.02, 
        max_iter_df=50
    ):
        """
        :param delta: controls the desired fooling rate (default = 80% fooling rate)
        :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
        :param epsilon: controls the l_p magnitude of the perturbation (default = 10/255.0)
        :param p_norm: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter_df: maximum number of iterations for deepfool (default = 10)
        :return: the universal perturbation.
        """
        
        ms.set_context(device_target="GPU")
        self.model.set_train(False)
        
        v = ms.Tensor(np.zeros_like(self.trainset[0][0]), ms.float32)
        fooling_rate = 0.0
        total_num = len(self.trainset)
        num_images = min(total_num, self.num_samples)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        order = np.array(tmp_list[:num_images])

        itr = 0
        while fooling_rate < 1-delta and itr < max_iter_uni:
            np.random.shuffle(order)
            print('Starting pass number ', itr)
            for idx, k in enumerate(order):
                cur_img, _ = self.trainset[k]
                cur_img = ms.Tensor(cur_img, ms.float32)
                perturb_img = cur_img + v

                orig_pred = self.model(cur_img.expand_dims(0)).asnumpy().argmax(axis=1)[0]
                pert_pred = self.model(perturb_img.expand_dims(0)).asnumpy().argmax(axis=1)[0]
                if orig_pred == pert_pred:
                    print('>> k = ', idx, ', pass #', itr)
                    dr, iterr, _, _ = self.deepfool_target(perturb_img, num_classes=num_classes,
                                                           overshoot=overshoot, max_iter=max_iter_df)
                    dr = ms.Tensor(dr.squeeze(), ms.float32)
                    if iterr < max_iter_df-1:
                        v = v + dr
                        v = self.proj_lp(v, epsilon, p_norm)

            itr += 1

            # 测试fooling rate
            test_num_images = 0
            est_labels_orig = []
            est_labels_pert = []
            
            self.test_loader = self.testset.to_generator_dataset(
                batch_size=200,
                shuffle=False,
                num_parallel_workers=1
            )
            
            for _, (images, _) in enumerate(self.test_loader):
                # input is (B, C, H, W)
                inputs = images
                inputs_pert = inputs + v.expand_dims(0)
                
                # output is (B, 10)
                outputs = self.model(inputs)
                outputs_perturb = self.model(inputs_pert)
                est_labels_orig += list(outputs.asnumpy().argmax(axis=1))
                est_labels_pert += list(outputs_perturb.asnumpy().argmax(axis=1))
                test_num_images += inputs.shape[0]
            fooling_rate = float(np.sum(np.array(est_labels_orig) != np.array(est_labels_pert))) / float(test_num_images)
            print('FOOLING RATE = ', fooling_rate)
        print('Final FOOLING RATE = ', fooling_rate)
        return v
    

class TUAP(Base):
    """
    Constrct poisoned datasets with TUAP method.
    
    Args:
    
    """
    def __init__(
        self,
        train_dataset,
        test_dataset,
        model,
        loss,
        
        benign_model,
        y_target,
        poisoned_rate,
        epsilon=0.031,
        delta=0.2,
        max_iter_uni=20,
        p_norm=np.inf,
        num_classes=10,
        overshoot=0.02,
        max_iter_df=50,
        p_samples=0.01,
        mask=None,  # can be none
        pattern=None,

        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        schedule=None,
        seed=0,
    ):
        
        self.y_target = y_target
        
        if mask is None:
            self.mask = np.ones((3, 32, 32), dtype=np.float32)
            self.mask = ms.Tensor(self.mask, ms.float32)
        else:
            if isinstance(mask, np.ndarray):
                self.mask = ms.Tensor(mask, ms.float32)
            else:
                raise ValueError("mask must be a numpy array")
            
            self.mask = mask
                
        if pattern is None:
            UAP_ins = UAP(
                benign_model,
                train_dataset,
                test_dataset,
                self.y_target,
                self.mask,
                p_samples=p_samples,
            )
            
            self.pattern = UAP_ins.universal_perturbation(
                delta=delta,
                max_iter_uni=max_iter_uni,
                epsilon=epsilon,
                p_norm=p_norm,
                num_classes=num_classes,
                overshoot=overshoot,
                max_iter_df=max_iter_df,
            )
            
            # save the pattern
            pattern_np = self.pattern.asnumpy()
            np.save('core/utils/uap/pattern.npy', pattern_np)
            
        else:
            self.pattern = pattern
        
        self.mask = self.mask.asnumpy()
        # if pattern is ms.Tensor, convert to np.ndarray
        if isinstance(self.pattern, ms.Tensor):
            self.pattern = self.pattern.asnumpy()
        
        
        poisoned_train_dataset = PoisonedDatasetFolder(
            benign_dataset=train_dataset,
            y_target=y_target,
            is_train_set=True,
            poisoned_rate=poisoned_rate,
            pattern=self.pattern,
            mask=self.mask,
            poisoned_transform_index=poisoned_transform_train_index,
            poisoned_target_transform_index=poisoned_target_transform_index,
        )
        
        poisoned_test_dataset = PoisonedDatasetFolder(
            benign_dataset=test_dataset,
            y_target=y_target,
            is_train_set=False,
            poisoned_rate=1.0,
            pattern=self.pattern,
            mask=self.mask,
            poisoned_transform_index=poisoned_transform_test_index,
            poisoned_target_transform_index=poisoned_target_transform_index,
        )
        
        super(TUAP, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            poisoned_train_dataset=poisoned_train_dataset,
            poisoned_test_dataset=poisoned_test_dataset,
            seed=seed
        )