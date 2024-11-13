"""
This module processes data.
"""
import os
import random
import hydra
from PIL import Image
import mindspore as ms
from mindspore import ops
from mindspore.dataset import transforms, vision
import numpy as np


class CustomData:
    """CustomData class, which is used to process data."""
    def __init__(self, data_dir, dataset_name, number_data_points):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.num_data = number_data_points
        self.extract_mean_std()

    def get_data_cfg(self):
        with hydra.initialize(config_path='breaching/config/case/data', version_base='1.1'):
            cfg = hydra.compose(config_name=self.dataset_name)
        return cfg

    def extract_mean_std(self):
        cfg = self.get_data_cfg()
        self.mean = ms.Tensor(list(cfg.mean))[None, :, None, None]
        self.std = ms.Tensor(list(cfg.std))[None, :, None, None]

    def process_data(self):
        """process data."""
        trans = transforms.Compose(
            [
                vision.Resize(size=(224)),
                vision.Rescale(1.0 / 255.0, 0),
                vision.HWC2CHW(),
            ]
        )
        file_name_li = os.listdir(self.data_dir)
        file_name_list = sorted(file_name_li, key=lambda x: int(x.split('-')[0]))
        assert len(file_name_list) >= int(self.num_data)
        imgs = []
        labels_ = []
        random.shuffle(file_name_list)
        for file_name in file_name_list[0:int(self.num_data)]:
            img = Image.open(self.data_dir+file_name).convert("RGB")
            tmp_img = ms.Tensor(trans(img)[0])
            imgs.append(tmp_img[None, :])
            label = int(file_name.split('-')[0])
            labels_.append(label)
        imgs = ops.concat(imgs, 0)
        labels = ms.Tensor(labels_)
        inputs = (imgs-self.mean)/self.std
        return dict(inputs=inputs, labels=labels)

    def save_recover(self, recover, original=None, save_pth='', sature=False):
        """save recovered data."""
        if original is not None:
            if isinstance(recover, dict):
                recover_imgs = ops.clip_by_value(recover['data']*self.std+self.mean, 0, 1)
                if sature:
                    recover_imgs = vision.AdjustSaturation(saturation_factor=sature)(recover_imgs)
                origina_imgs = ops.clip_by_value(original['data'] * self.std + self.mean, 0, 1)
                all_img = ops.concat([recover_imgs, origina_imgs], 2)
            else:
                recover_imgs = ops.clip_by_value(recover * self.std + self.mean, 0, 1)
                if sature:
                    recover_imgs = vision.AdjustSaturation(saturation_factor=sature)(recover_imgs)
                origina_imgs = ops.clip_by_value(original['data'] * self.std + self.mean, 0, 1)
                all_img = ops.concat([recover_imgs, origina_imgs], 2)
        else:
            if isinstance(recover, dict):
                recover_imgs = ops.clip_by_value(recover['data'] * self.std + self.mean, 0, 1)
                if sature:
                    recover_imgs = vision.AdjustSaturation(saturation_factor=sature)(recover_imgs)
                all_img = recover_imgs
            else:
                recover_imgs = ops.clip_by_value(recover * self.std + self.mean, 0, 1)
                if sature:
                    recover_imgs = vision.AdjustSaturation(saturation_factor=sature)(recover_imgs)
                all_img = recover_imgs
        self.save_array_img(all_img, save_pth)

    def save_array_img(self, img_4d, save_pth):
        # img_4d: msTensor n*3*H*W, 0-1
        imgs = img_4d.asnumpy()*225
        all_imgs = [im for im in imgs]
        tmp1 = np.concatenate(all_imgs, axis=-1)
        tmp2 = np.uint8(np.transpose(tmp1, [1, 2, 0]))
        Image.fromarray(tmp2, mode='RGB').save(save_pth)
