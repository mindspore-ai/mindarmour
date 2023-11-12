# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""main pipeline."""
import os
from PIL import Image
import cv2
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.vision as vision

from src.yolo import YOLOv3


context.set_context(mode=0, device_target="CPU", device_id=0)

TRAIN_DATA_PATH = "./car_dataset/train/images"
PATCH_SIZE = 100
EPOCH = 2
STEP_SIZE = 0.1
ATTACK_ITERS = 2

np.random.seed(2333)


class GradCam(nn.Cell):
    """Grad cam."""
    def __init__(self, model):
        super(GradCam, self).__init__()

        self.model_fea = model.feature_map
        self.head1 = model.detect_1
        self.head2 = model.detect_2

    def loss_fn(self, fea1, fea2, in_shape):
        output_big = self.head1(fea1, in_shape)
        output_small = self.head2(fea2, in_shape)
        loss = output_big[..., 7].sum() + output_small[..., 7].sum()
        return loss

    def get_feature_and_weights(self, img, img_shape):

        big_object_output, small_object_output = self.model_fea(img)
        grad_fn = ms.ops.value_and_grad(self.loss_fn, (0, 1))
        weights = grad_fn(big_object_output, small_object_output, img_shape)
        mean = ms.ops.ReduceMean()
        return (
            big_object_output,
            small_object_output,
            mean(weights[1][0], (2, 3)),
            mean(weights[1][1], (2, 3)),
        )

    def construct(self, model_input, in_shape):
        """Calculate grad cam."""
        fea1, fea2, weights1, weights2 = self.get_feature_and_weights(model_input, in_shape)

        cam1 = (weights1.expand_dims(-1).expand_dims(-1) * fea1).squeeze().sum(0)
        cam2 = (weights2.expand_dims(-1).expand_dims(-1) * fea2).squeeze().sum(0)

        cam1 = cam1 * (cam1 > 0) / cam1.max()
        cam2 = cam2 * (cam2 > 0) / cam2.max()

        return [cam1, cam2]


class VisualCAM:
    """Show grad cam image."""
    def __init__(self, model):
        self.grad_cam = GradCam(model=model)
        self.log_dir = "./"

    def __call__(self, img, img_shape):
        raw_img = img.asnumpy()[0].transpose((1, 2, 0))
        cam_map = self.grad_cam(img, img_shape)
        self.show_cam_on_image(raw_img, cam_map[0], img_id=0)
        self.show_cam_on_image(raw_img, cam_map[1], img_id=1)

    def show_cam_on_image(self, img, cam_map, img_id=0):
        """Show image grad cam."""
        mask_np = cam_map.asnumpy()

        heatmap = cv2.applyColorMap(np.uint8(255 * mask_np), cv2.COLORMAP_JET)[
            :, :, ::-1
        ]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.float32(heatmap) / 255
        cam_m = np.float32(img) + heatmap
        cam_m = cam_m / np.max(cam_m)
        Image.fromarray(np.uint8(255 * cam_m)).save(
            os.path.join(self.log_dir, "test_cam_%d.jpg" % img_id)
        )
        Image.fromarray(np.uint8(255 * heatmap)).save(
            os.path.join(self.log_dir, "cam_%d.jpg" % img_id)
        )


def submatrix(arr):
    x, y = np.nonzero(arr)
    return arr[x.min() : x.max() + 1, y.min() : y.max() + 1]


def init_patch_square(patch_size):
    patch_ini = np.expand_dims(
        vision.ToTensor()(
            Image.open("./patch_seed/cat.png")
            .convert("RGB")
            .resize((patch_size, patch_size))
        ),
        0,
    )
    return patch_ini, patch_ini.shape


def patch_transform(pattern, data_shape, input_patch_shape):
    """Transform adversarial patch for different samples."""
    # get dummy image
    x = np.zeros(data_shape)
    # get shape
    m_size = input_patch_shape[-1]

    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(4)
        for j_idx in range(pattern[i].shape[0]):
            pattern[i][j_idx] = np.rot90(pattern[i][j_idx], rot)
        # random location
        random_x = np.random.choice(x.shape[-2])
        if random_x + m_size > x.shape[-2]:
            while random_x + m_size > x.shape[-2]:
                random_x = np.random.choice(x.shape[-2])
        random_y = np.random.choice(x.shape[-1])
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(x.shape[-1])

        # apply patch to dummy image
        x[i][0][
            random_x : random_x + input_patch_shape[-1], random_y : random_y + input_patch_shape[-1]
        ] = pattern[i][0]
        x[i][1][
            random_x : random_x + input_patch_shape[-1], random_y : random_y + input_patch_shape[-1]
        ] = pattern[i][1]
        x[i][2][
            random_x : random_x + input_patch_shape[-1], random_y : random_y + input_patch_shape[-1]
        ] = pattern[i][2]

    masks = np.copy(x)
    masks[masks != 0] = 1.0
    return x, masks


def loss_graph(img):
    cam1, cam2 = cam(img, input_shape)
    loss = cam1.sum() / (
        (cam1.shape[0] * cam1.shape[1]) - (cam1 > 0).sum()
    ) + cam2.sum() / ((cam2.shape[0] * cam2.shape[1]) - (cam2 > 0).sum())
    return loss


def loss_smooth(img, matrix):
    s1 = ms.ops.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = ms.ops.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    matrix_mask = matrix[:, :, :-1, :-1]
    return (matrix_mask * (s1 + s2)).sum()


def loss_content(img, img_ori, canny):
    return (
        0.9 * (canny * ms.ops.pow(img - img_ori, 2)).sum()
        + 0.1 * ((1 - canny) * ms.ops.pow(img - img_ori, 2)).sum()
    )


def loss_sum(img, img_ori, m, canny):
    return (
        loss_graph(img)
        + 0.0001 * loss_smooth(img, m)
        + 0.0001 * loss_content(img, img_ori, canny)
    )


def attack(x, adv_patch, input_mask, iters=25):
    """Generate attack sample."""
    adv_x = x
    adv_x = ms.ops.mul((1 - input_mask), adv_x) + ms.ops.mul(input_mask, adv_patch)
    adv_x = ms.ops.clip_by_value(adv_x, 0, 1)
    adv_x_np = np.uint8(255 * adv_x.asnumpy()[0].transpose((1, 2, 0)))
    cv2.imwrite("adv_x.png", adv_x_np)
    canny = Tensor(cv2.Canny(adv_x_np, 128, 200)) >= 1

    adv_x = ms.Parameter(Tensor(adv_x.asnumpy(), ms.float32), requires_grad=True)
    adv_x_ori = ms.Parameter(Tensor(adv_x.asnumpy(), ms.float32), requires_grad=True)
    count = 0

    while True:
        count += 1
        grad_fn = ms.ops.value_and_grad(loss_sum)
        loss, adv_grad = grad_fn(adv_x, adv_x_ori, input_mask, canny)
        print("Loss:", loss)

        adv_patch = adv_patch - STEP_SIZE * adv_grad / adv_grad.max()
        adv_x = ms.ops.mul((1 - input_mask), adv_x) + ms.ops.mul(input_mask, adv_patch)
        adv_x = ms.ops.clip_by_value(adv_x, 0, 1)

        if count >= iters:
            break

        adv_x = ms.Parameter(Tensor(adv_x.asnumpy(), ms.float32), requires_grad=True)

    return adv_x, input_mask, adv_patch


class Iterable:
    """Iterable dataset."""
    def __init__(self, img_path):
        self.img_path = img_path
        self.imgs = os.listdir(img_path)

    def __getitem__(self, index):
        return vision.ToTensor()(
            Image.open(os.path.join(self.img_path, self.imgs[index])).convert("RGB")
        )

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    data = Iterable(TRAIN_DATA_PATH)
    dataset = GeneratorDataset(source=data, column_names=["data"])
    dataset = dataset.batch(batch_size=1)

    network = YOLOv3(is_training=False)
    ckpt_path = "./yolov3tiny_ascend_v190_coco2017_research_cv_mAP17.5_AP50acc36.0.ckpt"
    param_dict = load_checkpoint(ckpt_path)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith("moments."):
            continue
        elif key.startswith("yolo_network."):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values

    load_param_into_net(network, param_dict_new)

    network.set_train(False)

    cam = GradCam(network)

    patch, patch_shape = init_patch_square(patch_size=PATCH_SIZE)
    for e in range(EPOCH):
        for idx, img_data in enumerate(dataset.create_tuple_iterator()):
            input_shape = Tensor(
                (img_data[0].shape[2], img_data[0].shape[3]), mstype.float32
            )
            patch, mask = patch_transform(patch, img_data[0].shape, patch_shape)
            patch, mask = Tensor(patch), Tensor(mask)
            x_adv, mask, patch = attack(
                img_data[0], patch, mask, iters=ATTACK_ITERS
            )
            masked_patch = ms.ops.mul(mask, patch)
            patch_ori = masked_patch.asnumpy()
            new_patch = np.zeros(patch_shape)
            for j in range(new_patch.shape[0]):
                for k in range(new_patch.shape[1]):
                    new_patch[j][k] = submatrix(patch_ori[j][k])
            patch = new_patch

    np.save("patch.npy", patch)
