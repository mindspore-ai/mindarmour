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
"""Yolo eval."""
import datetime
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import tqdm
import mindspore as ms
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset, vision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model_utils.config import config
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.moxing_adapter import moxing_wrapper
from src.logger import get_logger
from src.yolo import YOLOv3


class Redirct:
    """Redirct"""
    def __init__(self):
        self.content = ""

    def write(self, content):
        """write"""
        self.content += content

    def flush(self):
        """flush"""
        self.content = ""


class DetectionEngine:
    """Detection engine."""
    def __init__(self, config_detection):
        self.eval_ignore_threshold = config_detection.eval_ignore_threshold
        if config.finetune:
            self.labels = [
                'without_mask', 'with_mask', 'mask_weared_incorrect'
            ]
        else:
            self.labels = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush'
            ]
        self.num_classes = len(self.labels)
        self.results = {}
        self.file_path = ''
        self.save_prefix = config_detection.outputs_dir
        self.det_boxes = []
        self.nms_thresh = config_detection.nms_thresh
        self.multi_label = config_detection.multi_label
        self.multi_label_thresh = config_detection.multi_label_thresh
        self.coco_cat_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]
        with open("./coco_val_2017/annotations/instances_val2017.json",
                  "r",
                  encoding="utf-8") as f:
            content = json.load(f)
        self.content = content['annotations']
        self.ann_file = "./coco_val_2017/annotations/instances_val2017.json"

    def do_nms_for_results(self):
        """Get result boxes."""
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._nms(dets, self.nms_thresh)

                keep_box = [{
                    'image_id': int(img_id),
                    'category_id': int(clsi),
                    'bbox': list(dets[i][:4].astype(float)),
                    'score': dets[i][4].astype(float)
                } for i in keep_index]
                self.det_boxes.extend(keep_box)

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] -
                                    intersect_area)

            indexes = np.where(ovr <= threshold)[0]
            order = order[indexes + 1]
        return reserved_boxes

    def _diou_nms(self, dets, thresh=0.5):
        """
        convert xywh -> xmin ymin xmax ymax
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1)**2 + (center_y2 -
                                                       center_y1)**2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x)**2 + (out_max_y -
                                                       out_min_y)**2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def write_result(self):
        """Save result to file."""
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.det_boxes, f, indent=2)
        except IOError as e:
            raise RuntimeError(
                "Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def get_eval_result(self):
        """Get eval result."""
        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(self.file_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        coco_eval.summarize()
        m_ap = coco_eval.stats[0]
        print("\n======================================\n")
        print(f"mAP: {m_ap}")
        sys.stdout = stdout
        return rdct.content

    def detect(self, outputs, batch, image_shape, image_id):
        """Detect boxes."""
        outputs_num = len(outputs)
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_id in range(outputs_num):
                # 32, 52, 52, 3, 85
                out_item = outputs[out_id]
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                # get number of items in one head, [B, gx, gy, anchors, 5+80]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape
                img_id = int(image_id)
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]
                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1),
                                            axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                cls_emb = cls_emb.reshape(-1, self.num_classes)
                if self.multi_label:
                    conf = conf.reshape(-1, 1)
                    # create all False
                    confidence = cls_emb * conf
                    flag = cls_emb > self.multi_label_thresh
                    flag = flag.nonzero()
                    for index in range(len(flag[0])):
                        i = flag[0][index]
                        j = flag[1][index]
                        confi = confidence[i][j]
                        if confi < self.eval_ignore_threshold:
                            continue
                        if img_id not in self.results:
                            self.results[img_id] = defaultdict(list)
                        x_lefti = max(0, x_top_left[i])
                        y_lefti = max(0, y_top_left[i])
                        wi = min(w[i], ori_w)
                        hi = min(h[i], ori_h)
                        clsi = j
                        # transform catId to match coco
                        coco_clsi = self.coco_cat_ids[clsi]
                        self.results[img_id][coco_clsi].append(
                            [x_lefti, y_lefti, wi, hi, confi])
                else:
                    cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1),
                                                axis=-1)
                    conf = conf.reshape(-1)
                    cls_argmax = cls_argmax.reshape(-1)

                    # create all False
                    flag = np.random.random(cls_emb.shape) > sys.maxsize
                    for i in range(flag.shape[0]):
                        c = cls_argmax[i]
                        flag[i, c] = True
                    confidence = cls_emb[flag] * conf

                    for x_lefti, y_lefti, wi, hi, confi, clsi in zip(
                            x_top_left, y_top_left, w, h, confidence,
                            cls_argmax):
                        if confi < self.eval_ignore_threshold:
                            continue
                        if img_id not in self.results:
                            self.results[img_id] = defaultdict(list)
                        x_lefti = max(0, x_lefti)
                        y_lefti = max(0, y_lefti)
                        wi = min(wi, ori_w)
                        hi = min(hi, ori_h)
                        # transform catId to match coco
                        coco_clsi = self.coco_cat_ids[clsi]
                        self.results[img_id][coco_clsi].append(
                            [x_lefti, y_lefti, wi, hi, confi])


def convert_testing_shape(config_testing_shape):
    """Convert testing shape to list."""
    testing_shape = [int(config_testing_shape), int(config_testing_shape)]
    return testing_shape


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(
                os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(
                            int(i * 100 / data_num)),
                              flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(
                    int((time.time() - s_time) / 60),
                    int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path,
                                  config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(),
                                 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(
            get_device_id(), zip_file_1, save_dir_1))

    config.log_path = os.path.join(config.output_path, config.log_path)


def patch_transform(patch, data_shape, patch_shape):
    """Transform adv patch"""

    # get dummy image
    x = np.zeros(data_shape)
    # get shape
    m_size = patch_shape[-1]

    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(4)
        for k in range(patch[i].shape[0]):
            patch[i][k] = np.rot90(patch[i][k], rot)
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
        x[i][0][random_x:random_x + patch_shape[-1],
                random_y:random_y + patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x + patch_shape[-1],
                random_y:random_y + patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x + patch_shape[-1],
                random_y:random_y + patch_shape[-1]] = patch[i][2]

    mask = np.copy(x)
    mask[mask != 0] = 1.0
    return x, mask


def _reshape_data(image, image_size):
    """Reshape data"""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    ori_w, ori_h = image.size
    ori_image_shape = np.array([ori_w, ori_h], np.int32)
    # original image shape fir:H sec:W
    h, w = image_size
    interp = get_interp_method(interp=9, sizes=(ori_h, ori_w, h, w))
    image = image.resize((w, h), pil_image_reshape(interp))
    image_data = statistic_normalize_img(image, statistic_norm=True)
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = np.concatenate([image_data, image_data, image_data],
                                    axis=-1)
    image_data = image_data.astype(np.float32)
    return image_data, ori_image_shape


def statistic_normalize_img(img, statistic_norm):
    """Statistic normalize images."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if statistic_norm:
        img = (img - mean) / std
    return img


def get_interp_method(interp, sizes=()):
    """
    Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic or Bilinear.

    Args:
        interp (int): Interpolation method for all resizing operations.

            - 0: Nearest Neighbors Interpolation.
            - 1: Bilinear interpolation.
            - 2: Bicubic interpolation over 4x4 pixel neighborhood.
            - 3: Nearest Neighbors. Originally it should be Area-based, as we cannot find Area-based,
              so we use NN instead. Area-based (resampling using pixel area relation).
              It may be a preferred method for image decimation, as it gives moire-free results.
              But when the image is zoomed, it is similar to the Nearest Neighbors method. (used by default).
            - 4: Lanczos interpolation over 8x8 pixel neighborhood.
            - 9: Cubic for enlarge, area for shrink, bilinear for others.
            - 10: Random select from interpolation method mentioned above.

        sizes (tuple): Format should like (old_height, old_width, new_height, new_width),
            if None provided, auto(9) will return Area(2) anyway. Default: ()

    Returns:
        int, interp method from 0 to 4.
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            if nh < oh and nw < ow:
                return 0
            return 1
        return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def pil_image_reshape(interp):
    """Reshape pil image."""
    reshape_type = {
        0: Image.NEAREST,
        1: Image.BILINEAR,
        2: Image.BICUBIC,
        3: Image.NEAREST,
        4: Image.LANCZOS,
    }
    return reshape_type[interp]


class Iterable:
    """Iterable dataset"""
    def __init__(self, img_path):
        self.img_path = img_path
        self.imgs = os.listdir(img_path)
        self.mean = [m * 255 for m in [0.485, 0.456, 0.406]]
        self.std = [s * 255 for s in [0.229, 0.224, 0.225]]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path,
                                      self.imgs[index])).convert('RGB')
        img, ori_image_shape = _reshape_data(img, image_size=(640, 640))
        img = vision.HWC2CHW()(img)
        img_id = int(self.imgs[index].split('.')[0])
        return img, img_id, ori_image_shape

    def __len__(self):
        return len(self.imgs)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_test():
    """The function of eval."""
    start_time = time.time()

    config.data_root = "./coco_val_2017/val2017"
    device_id = 0
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target,
                        device_id=device_id)

    # logger
    config.outputs_dir = os.path.join(
        './results',
        datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(
        os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    config.logger = get_logger(config.outputs_dir, rank_id)

    config.logger.info('Creating Network....')
    network = YOLOv3(is_training=False)

    config.logger.info(config.pretrained)
    if os.path.isfile(config.pretrained):
        param_dict = ms.load_checkpoint(config.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        ms.load_param_into_net(network, param_dict_new)
        config.logger.info('load_model %s success', config.pretrained)
    else:
        config.logger.info('%s not exists or not a pre-trained file',
                           config.pretrained)
        assert FileNotFoundError('%s not exists or not a pre-trained file',
                                 config.pretrained)
        exit(1)

    data_root = config.data_root
    data = Iterable(data_root)
    dataset = GeneratorDataset(
        source=data, column_names=["image", "image_id", "image_shape"])
    dataset = dataset.batch(batch_size=1)

    network.set_train(False)
    detection = DetectionEngine(config)

    input_shape = Tensor((640, 640))

    config.logger.info('Start inference....')
    for data in tqdm.tqdm(dataset.create_dict_iterator()):

        image = data["image"].asnumpy()

        image = Tensor(image)
        image_shape_ = data["image_shape"][0]
        image_id_ = data["image_id"].asnumpy()

        if config.attack_eval:
            patch = np.load('patch.npy')
            patch, mask = patch_transform(patch, image.shape, patch.shape)
            patch, mask = Tensor(patch, ms.float32), Tensor(mask, ms.float32)
            image = ms.ops.mul((1 - mask), image) + ms.ops.mul(mask, patch)
            image = ms.ops.clip_by_value(image, 0, 1)

        prediction = network(image, input_shape)
        output_big, output_small = prediction
        output_big = output_big.asnumpy()
        output_small = output_small.asnumpy()
        image_shape_ = image_shape_.asnumpy()
        detection.detect([output_small, output_big], config.per_batch_size,
                         image_shape_, image_id_)

        image = image.asnumpy()
        Image.fromarray(np.uint8(image[0].transpose(1, 2, 0) *
                                 255)).save('test_img.png')

    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    config.logger.info('result file path: %s', result_file_path)

    eval_result = detection.get_eval_result()
    cost_time = time.time() - start_time
    config.logger.info('\n=============eval reulst=========\n %s', eval_result)
    config.logger.info('testing cost time {:.2f}h'.format(cost_time / 3600.))


if __name__ == "__main__":

    run_test()
