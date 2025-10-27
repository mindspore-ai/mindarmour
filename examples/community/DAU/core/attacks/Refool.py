
'''
This is the simplified version of Refool attack [1], where the reflection images are (randomly) given instead of by optimization. 
Note: it is under the poison-label instead of the clean-label mode since it has minor effects under the clean-label settings. 

Reference:
[1] Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks. ECCV 2020.
'''
# sys
import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# mindspore
import mindspore
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import cosine_decay_lr
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset
import mindspore.ops as ops
import mindspore.numpy as mnp
# core
from core.attacks import Attack
from ..Base import Base
# numpy
import numpy as np
import cv2
import time
import random
from PIL import Image
from copy import deepcopy
from scipy import stats
from tqdm import tqdm
#utils
from utils import compute_accuracy, read_image


def save_dataset(dataset, poison_datasets_path):

    np.savez(
        poison_datasets_path,
        data=dataset.data,
        targets=dataset.targets,
        classes = dataset.classes,
        y_target=dataset.y_target,
        poisoned_rate=dataset.poisoned_rate,
        poisoned_set = dataset.poisoned_set,
        poison_indices =dataset.poison_indices,
        modified_targets=dataset.modified_targets
    )

def load_dataset(poison_datasets_path):

    data_dict = np.load(poison_datasets_path, allow_pickle=True)

    # load reflection images
    # print(f"BASE_DIR:{BASE_DIR}\n")
    reflection_images = []
    reflection_data_dir = os.path.join(BASE_DIR,"datasets/VOCdevkit/VOC2012/JPEGImages/")
    reflection_image_path = os.listdir(reflection_data_dir)
    reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]

    # print(f"reflection_images:{reflection_images[0].shape}")
  
    dataset = PoisonedVisionDataset(
        benign_dataset=[], 
        y_target=int(data_dict['y_target']),
        poisoned_rate=float(data_dict['poisoned_rate']),
        reflection_candidates=reflection_images
    )

    dataset.data = data_dict['data']
    dataset.targets = data_dict['targets']
    dataset.classes = data_dict['classes']
    dataset.poisoned_set = data_dict['poisoned_set']
    dataset.poison_indices = data_dict['poison_indices']
    dataset.modified_targets = data_dict['modified_targets']

    return dataset


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size

    _, pred = ops.TopK(sorted=True)(output, maxk)
    pred = pred.T
    correct = ops.Equal()(pred, target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype('float32').sum(0)
        res.append(correct_k * (100.0 / batch_size))
    return res

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

class AddTriggerMixin(object):
    """Add reflection-based trigger to images.

    Args:
        total_num (integer): number of images in the dataset
        reflection_cadidates (List of numpy.ndarray of shape (H, W, C) or (H, W))
        max_image_size (integer): max(Height, Weight) of returned image
        ghost_rate (float): rate of ghost reflection
        alpha_b (float): the ratio of background image in blended image, alpha_b should be in $(0,1)$, set to -1 if random alpha_b is desired
        offset (tuple of 2 interger): the offset of ghost reflection in the direction of x axis and y axis, set to (0,0) if random offset is desired
        sigma (interger): the sigma of gaussian kernel, set to -1 if random sigma is desired
        ghost_alpha (interger): ghost_alpha should be in $(0,1)$, set to -1 if random ghost_alpha is desire
    """
    def __init__(self, total_num, reflection_candidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        super(AddTriggerMixin,self).__init__()
        self.reflection_candidates = reflection_candidates
        self.max_image_size=max_image_size
        # generate random numbers for refelection-based trigger generation and keep them fixed during training 
        self.reflection_candidates_index = np.random.randint(0,len(self.reflection_candidates),total_num)
        self.alpha_bs = 1.-np.random.uniform(0.05,0.45,total_num) if alpha_b<0 else np.zeros(total_num)+alpha_b
        self.ghost_values = (np.random.uniform(0,1,total_num) < ghost_rate)
        if offset == (0,0):
            self.offset_xs = np.random.random_integers(3,8,total_num)
            self.offset_ys = np.random.random_integers(3,8,total_num)
        else:
            self.offset_xs = np.zeros((total_num,),np.int32) + offset[0]
            self.offset_ys = np.zeros((total_num,),np.int32) + offset[1]
        self.ghost_alpha = ghost_alpha
        self.ghost_alpha_switchs = np.random.uniform(0,1,total_num)
        self.ghost_alphas = np.random.uniform(0.15,0.5,total_num) if ghost_alpha < 0 else np.zeros(total_num)+ghost_alpha
        self.sigmas = np.random.uniform(1,5,total_num) if sigma<0 else np.zeros(total_num)+sigma
        self.atts = 1.08 + np.random.random(total_num)/10.0
        self.new_ws = np.random.uniform(0,1,total_num)
        self.new_hs = np.random.uniform(0,1,total_num)

    def _add_trigger(self, sample, index):
        """Add reflection-based trigger to images.

        Args:        
            sample (torch.Tensor): shape (C,H,W),
            index (interger): index of sample in original dataset
        """
        # img_b = sample.permute(1,2,0).numpy() # background
        img_b = sample.numpy()
        img_r = self.reflection_candidates[self.reflection_candidates_index[index]] # reflection

        # print(f"img_b:{img_b.shape},img_r:{img_r.shape}\n")
        # print(f"img_b:{img_b},img_r:{img_r}\n")

        h, w, channels = img_b.shape
        if channels == 1 and img_r.shape[-1]==3: 
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]

        b = np.float32(img_b) / 255.
        r = np.float32(img_r) / 255.
        
        # convert t.shape to max_image_size's limitation
        scale_ratio = float(max(h, w)) / float(self.max_image_size)
        w, h = (self.max_image_size, int(round(h / scale_ratio))) if w > h \
            else (int(round(w / scale_ratio)), self.max_image_size)
        b = cv2.resize(b, (w, h), cv2.INTER_CUBIC)
        r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)
        if channels == 1:
            b = b[:,:,np.newaxis]
            r = r[:,:,np.newaxis]
        
        alpha_b = self.alpha_bs[index]
        if self.ghost_values[index]:
            b = np.power(b, 2.2)
            r = np.power(r, 2.2)

            # generate the blended image with ghost effect
            offset = (self.offset_xs[index],self.offset_ys[index])
            r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                         'constant', constant_values=0)
            r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))
            ghost_alpha = self.ghost_alpha
            if ghost_alpha < 0:
                ghost_alpha_switch = 1 if self.ghost_alpha_switchs[index] > 0.5 else 0
                ghost_alpha = abs(ghost_alpha_switch - self.ghost_alphas[index])
            
            ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
            ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
            if channels==1:
                ghost_r = ghost_r[:,:,np.newaxis]
            reflection_mask = ghost_r * (1 - alpha_b)
            blended = reflection_mask + b * alpha_b
            transmission_layer = np.power(b * alpha_b, 1 / 2.2)

            ghost_r = np.power(reflection_mask, 1 / 2.2)
            ghost_r[ghost_r > 1.] = 1.
            ghost_r[ghost_r < 0.] = 0.

            blended = np.power(blended, 1 / 2.2)
            blended[blended > 1.] = 1.
            blended[blended < 0.] = 0.

            reflection_layer = np.uint8(ghost_r * 255)
            blended = np.uint8(blended * 255)
            transmission_layer = np.uint8(transmission_layer * 255)
        else:
            # generate the blended image with focal blur
            sigma = self.sigmas[index]

            b = np.power(b, 2.2)
            r = np.power(r, 2.2)

            sz = int(2 * np.ceil(2 * sigma) + 1)
            r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
            if channels==1:
                r_blur = r_blur[:,:,np.newaxis]
            blend = r_blur + b

            # get the reflection layers' proper range
            att = self.atts[index]
            for i in range(channels):
                maski = blend[:, :, i] > 1
                mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
                r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
            r_blur[r_blur >= 1] = 1
            r_blur[r_blur <= 0] = 0

            def gen_kernel(kern_len=100, nsig=1):
                """Returns a 2D Gaussian kernel array."""
                interval = (2 * nsig + 1.) / kern_len
                x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
                # get normal distribution
                kern1d = np.diff(stats.norm.cdf(x))
                kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
                kernel = kernel_raw / kernel_raw.sum()
                kernel = kernel / kernel.max()
                return kernel
            h, w = r_blur.shape[0: 2]
            new_w = int(self.new_ws[index]*(self.max_image_size - w - 10)) if w < self.max_image_size - 10 else 0
            new_h = int(self.new_hs[index]*(self.max_image_size - h - 10)) if h < self.max_image_size - 10 else 0

            g_mask = gen_kernel(self.max_image_size, 3)
            g_mask = np.dstack((g_mask, )*channels)
            alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_b / 2.)

            r_blur_mask = np.multiply(r_blur, alpha_r)
            blur_r = min(1., 4 * (1 - alpha_b)) * r_blur_mask
            blend = r_blur_mask + b * alpha_b

            transmission_layer = np.power(b * alpha_b, 1 / 2.2)
            r_blur_mask = np.power(blur_r, 1 / 2.2)
            blend = np.power(blend, 1 / 2.2)
            blend[blend >= 1] = 1
            blend[blend <= 0] = 0
            blended = np.uint8(blend * 255)

        return Tensor(blended.transpose(2,0,1), mindspore.float32)


class PoisonedVisionDataset(AddTriggerMixin):
    def __init__(self, benign_dataset, y_target, poisoned_rate, reflection_candidates,
                 max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        AddTriggerMixin.__init__(
            self,
            len(benign_dataset),
            reflection_candidates,
            max_image_size,
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha)

        def convert_to_numpy(dataset):
            data_list = []
            label_list = []
            for data in dataset.create_dict_iterator():
                data_list.append(data['image'].asnumpy())
                label_list.append(data['label'].asnumpy())

            return np.array(data_list), np.array(label_list)

        if len(benign_dataset) > 0:
            self.data, self.targets = convert_to_numpy(benign_dataset)
        else:
            self.data = []
            self.targets = []

        self.dataset = benign_dataset
        self.classes = np.array(list(set(self.targets)))
   
        self.y_target = y_target
        self.poisoned_rate = poisoned_rate

        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'

        tmp_list = np.arange(len(self.data))[~np.array(self.targets == y_target)]
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        self.poison_indices = sorted(list(self.poisoned_set))

        for index in self.poison_indices:
            img = self.data[index]
            if img.shape[0] == 1:
                img = Image.fromarray(img.squeeze(), mode='L')
            elif img.shape[0] == 3:
                # print(f"img.shape:{img.shape},img:{img}\n")
                img_np = np.transpose(img, (1, 2, 0))
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8)
                img = Image.fromarray(img_np)
            elif len(img.shape) == 3 and img.shape[2] == 3: 
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
    
            # print(f"Tensor(np.array(img)):{Tensor(np.array(img))}\n")

            img = self._add_trigger(Tensor(np.array(img)), index)
            img = img.transpose(1, 2, 0).asnumpy().astype(np.uint8)
            resize_img = Image.fromarray(img).resize((32, 32))
            self.data[index] = np.transpose(np.array(resize_img), (2, 0, 1))
            
        self.modified_targets = np.array(deepcopy(self.targets))
        self.modified_targets[self.poison_indices] = y_target
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.modified_targets[index])
        img = img.squeeze()
        img = (img / 255.0).astype(np.float32)
        return img, target, index
    
    def __len__(self):
        return len(self.data)
    
    def get_real_targets(self):
        return self.targets
    
    def get_classes(self):
        return self.classes
    
    def get_y_target(self):
        return self.y_target
    
    def get_poisoning_rate(self):
        return self.poisoned_rate
    
    def get_poison_indices(self):
        return self.poison_indices
    
    def get_modified_targets(self):
        return self.modified_targets

def create_train_loader(dataset, batch_size, num_workers):
    def generator():
        for idx in range(len(dataset)):
            yield dataset[idx]  # 这会返回 (img, target, index)

    return GeneratorDataset(
        source=generator,
        column_names=["data", "label", "index"],
        shuffle=True,
        num_parallel_workers=1
    ).batch(batch_size, drop_remainder=False)

class Refool(Base, Attack):
    def __init__(self, task, attack_schedule):
        
        schedule = None
        if 'train_schedule' in attack_schedule:
            schedule = attack_schedule['train_schedule']

        Base.__init__(self, task, schedule=attack_schedule.get('train_schedule'))
        Attack.__init__(self)

        self.attack_schedule = attack_schedule
        assert 'attack_strategy' in self.attack_schedule, "Attack_config must contain 'attack_strategy' configuration!"
        self.attack_strategy = attack_schedule['attack_strategy']

        self.poisoned_train_dataset = None
        self.poisoned_test_dataset = None
    
    def get_attack_strategy(self):
        return self.attack_strategy
    
    def set_poisoned_dataset(self, poisoned_train_dataset=None, poisoned_test_dataset=None):
        self.poisoned_train_dataset, self.poisoned_test_dataset = poisoned_train_dataset, poisoned_test_dataset

    def create_poisoned_dataset(self, benign_dataset, y_target=None, poisoned_rate=None, train=True):
        if y_target is None:
            assert 'y_target' in self.attack_schedule, "Attack_config must contain 'y_target' configuration!"
            y_target = self.attack_schedule['y_target']
        if poisoned_rate is None:
            assert 'poisoned_rate' in self.attack_schedule, "Attack_config must contain 'poisoned_rate' configuration!"
            poisoned_rate = self.attack_schedule['poisoned_rate']

        assert 'reflection_candidates' in self.attack_schedule, "Attack_config must contain 'reflection_candidates' configuration!"
        reflection_candidates = self.attack_schedule['reflection_candidates']

        max_image_size = 560
        ghost_rate = 0.49
        alpha_b = -1.
        offset = (0, 0)
        sigma = -1
        ghost_alpha = -1.

        return PoisonedVisionDataset(benign_dataset, y_target, poisoned_rate, reflection_candidates, max_image_size, ghost_rate, alpha_b, offset, sigma, ghost_alpha)
    
    def train(self, train_dataset=None, test_dataset=None, schedule=None):
        
        self.poisoned_train_dataset = train_dataset

        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        train_loader = GeneratorDataset(
            train_dataset, 
            column_names=["data", "label", "index"],
            shuffle=True, 
            num_parallel_workers=1
        ).batch(self.current_schedule['batch_size'], drop_remainder=True)
        
        steps_per_epoch = len(train_dataset) // self.current_schedule['batch_size']
        total_steps =  self.current_schedule['epochs'] * steps_per_epoch

        lr_tensor = cosine_decay_lr(
            min_lr=0.001,
            max_lr=0.01,
            total_step=total_steps,
            step_per_epoch=steps_per_epoch,
            decay_epoch=self.current_schedule['epochs']
        )
      
        optimizer = self.optimizer(
            self.model.trainable_params(), 
            learning_rate=lr_tensor, 
            momentum=self.current_schedule['momentum'], 
            weight_decay=self.current_schedule['weight_decay']
        )
        
        # Define train step function
        def forward_fn(data, label):
            logits = self.model(data)
            loss = self.loss(logits, label)
            return loss, logits
        
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            return loss

        self.model.set_train(True)
        iteration = 0
        for i in range(self.current_schedule['epochs']):
            for step, (batch_img, batch_label,_) in enumerate(train_loader):
                
                batch_img = ops.cast(batch_img, ms.float32)
                batch_label = ops.cast(batch_label, ms.int32)

                loss = train_step(batch_img, batch_label)
                iteration += 1

                global_step = i * steps_per_epoch + step
                current_lr = lr_tensor[global_step]
                log_iteration_interval = self.current_schedule['log_iteration_interval']
                if iteration % log_iteration_interval == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{step + 1}\{len(train_dataset)//self.current_schedule['batch_size']}, lr:{current_lr}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    print(msg)


            # test_iteration_interval = self.current_schedule['test_epoch_interval']
            test_iteration_interval = 1
            if (i + 1) %  test_iteration_interval == 0:

                print("==========Test result on poisoned test dataset==========")

                predict_digits, labels = self._test(test_dataset, model=deepcopy(self.model))
        
                poisoned_test_indexs = test_dataset.get_poison_indices()
                benign_test_indexs = list(set(range(len(test_dataset))) - set(poisoned_test_indexs))
                poisoned_test_indexs = poisoned_test_indexs.tolist()
                
                benign_acc = compute_accuracy(predict_digits[benign_test_indexs], labels[benign_test_indexs], topk=(1,3,5))
                poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs], labels[poisoned_test_indexs], topk=(1,3,5))
                
                print(f"Total samples: {len(test_dataset)}, poisoning samples: {len(poisoned_test_indexs)}, benign samples: {len(benign_test_indexs)}")
                print(f"Benign_accuracy: {benign_acc[0]}, poisoning_accuracy: {poisoned_acc[0]}")

    def _test(self, dataset, model=None, batch_size=128, num_workers=8):
        if model is None:
            model = self.model
        else:
            model = model

        self.model.set_train(False)
        test_loader = create_train_loader(dataset, batch_size, num_workers)

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
        print(msg)
        
        last_time = time.time()
        predict_digits, labels = self._test(test_dataset, model=deepcopy(model), 
                                          batch_size=current_schedule['batch_size'], 
                                          num_workers=current_schedule['num_workers'])

        total_num = labels.shape[0]

        prec1, prec5 = compute_accuracy(predict_digits, labels, topk=(1, 5))
        top1_correct = int(round(prec1.item() / 100.0 * total_num))
        top5_correct = int(round(prec5.item() / 100.0 * total_num)) 
        
        msg = "\n==========Test result on test dataset==========\n"
        print(msg)
        msg = f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
        print(msg)
        
        return predict_digits, labels
