# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Train set"""
import os
import re
import numpy as np
import face_recognition as fr
import face_recognition_models as frm
import dlib
from PIL import Image, ImageDraw
import mindspore
import mindspore.dataset.vision.py_transforms as P
from mindspore.dataset.vision.py_transforms import ToPIL as ToPILImage
from mindspore.dataset.vision.py_transforms import ToTensor
from mindspore import Parameter, ops, nn, Tensor
from loss_design import MyTrainOneStepCell, MyWithLossCellTargetAttack, \
    MyWithLossCellNonTargetAttack, FaceLossTargetAttack, FaceLossNoTargetAttack


class FaceAdversarialAttack():
    """
    Class used to create adversarial facial recognition attacks.

    Args:
        input_img (numpy.ndarray): The input image.
        target_img (numpy.ndarray): The target image.
        seed (int): optional Sets custom seed for reproducibility. Default is generated randomly.
        net (mindspore.Model): face recognition model.
    """
    def __init__(self, input_img, target_img, net, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.mean = Tensor([0.485, 0.456, 0.406])
        self.std = Tensor([0.229, 0.224, 0.225])
        self.expand_dims = mindspore.ops.ExpandDims()
        self.imageize = ToPILImage()
        self.tensorize = ToTensor()
        self.normalize = P.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resnet = net
        self.input_tensor = Tensor(self.normalize(self.tensorize(input_img)))
        self.target_tensor = Tensor(self.normalize(self.tensorize(target_img)))
        self.input_emb = self.resnet(self.expand_dims(self.input_tensor, 0))
        self.target_emb = self.resnet(self.expand_dims(self.target_tensor, 0))
        self.adversarial_emb = None
        self.mask_tensor = create_mask(input_img)
        self.ref = self.mask_tensor
        self.pm = Parameter(self.mask_tensor)
        self.opt = nn.Adam([self.pm], learning_rate=0.01, weight_decay=0.0001)

    def train(self, attack_method):
        """
        Optimized adversarial image.

        Args:
            attack_method (String) : Including target attack and non_target attack.

        Returns:
            Tensor, adversarial image.
            Tensor, mask image.
        """

        if attack_method == "non_target_attack":
            loss = FaceLossNoTargetAttack()
            net_with_criterion = MyWithLossCellNonTargetAttack(self.resnet, loss, self.input_tensor)
        if attack_method == "target_attack":
            loss = FaceLossTargetAttack(self.target_emb)
            net_with_criterion = MyWithLossCellTargetAttack(self.resnet, loss, self.input_tensor)

        train_net = MyTrainOneStepCell(net_with_criterion, self.opt)

        for i in range(2000):

            self.mask_tensor = Tensor(self.pm)

            loss = train_net(self.mask_tensor)

            print("epoch %d ,loss: %f \n " % (i, loss.asnumpy().item()))

            self.mask_tensor = ops.clip_by_value(
                self.mask_tensor, Tensor(0, mindspore.float32), Tensor(1, mindspore.float32))

        adversarial_tensor = apply(
            self.input_tensor,
            (self.mask_tensor - self.mean[:, None, None]) / self.std[:, None, None],
            self.ref)

        adversarial_tensor = self._reverse_norm(adversarial_tensor)
        processed_input_tensor = self._reverse_norm(self.input_tensor)
        processed_target_tensor = self._reverse_norm(self.target_tensor)

        return {
            "adversarial_tensor": adversarial_tensor,
            "mask_tensor": self.mask_tensor,
            "processed_input_tensor": processed_input_tensor,
            "processed_target_tensor": processed_target_tensor
        }

    def test_target_attack(self):
        """
        The model is used to test the recognition ability of adversarial images under target attack.
        """

        adversarial_tensor = apply(
            self.input_tensor,
            (self.mask_tensor - self.mean[:, None, None]) / self.std[:, None, None],
            self.ref)

        self.adversarial_emb = self.resnet(self.expand_dims(adversarial_tensor, 0))
        self.input_emb = self.resnet(self.expand_dims(self.input_tensor, 0))
        self.target_emb = self.resnet(self.expand_dims(self.target_tensor, 0))

        adversarial_index = np.argmax(self.adversarial_emb.asnumpy())
        target_index = np.argmax(self.target_emb.asnumpy())
        input_index = np.argmax(self.input_emb.asnumpy())

        print("input_label:", input_index)
        print("target_label:", target_index)
        print("The confidence of the input image on the input label:", self.input_emb.asnumpy()[0][input_index])
        print("The confidence of the input image on the target label:", self.input_emb.asnumpy()[0][target_index])
        print("================================")
        print("adversarial_label:", adversarial_index)
        print("The confidence of the adversarial sample on the correct label:",
              self.adversarial_emb.asnumpy()[0][input_index])
        print("The confidence of the adversarial sample on the target label:",
              self.adversarial_emb.asnumpy()[0][target_index])
        print("input_label: %d, target_label: %d, adversarial_label: %d"
              % (input_index, target_index, adversarial_index))

    def test_non_target_attack(self):
        """
        The model is used to test the recognition ability of adversarial images under non_target attack.
        """

        adversarial_tensor = apply(
            self.input_tensor,
            (self.mask_tensor - self.mean[:, None, None]) / self.std[:, None, None],
            self.ref)

        self.adversarial_emb = self.resnet(self.expand_dims(adversarial_tensor, 0))
        self.input_emb = self.resnet(self.expand_dims(self.input_tensor, 0))

        adversarial_index = np.argmax(self.adversarial_emb.asnumpy())
        input_index = np.argmax(self.input_emb.asnumpy())

        print("input_label:", input_index)
        print("The confidence of the input image on the input label:", self.input_emb.asnumpy()[0][input_index])
        print("================================")
        print("adversarial_label:", adversarial_index)
        print("The confidence of the adversarial sample on the correct label:",
              self.adversarial_emb.asnumpy()[0][input_index])
        print("The confidence of the adversarial sample on the adversarial label:",
              self.adversarial_emb.asnumpy()[0][adversarial_index])
        print(
            "input_label: %d, adversarial_label: %d" % (input_index, adversarial_index))

    def _reverse_norm(self, image_tensor):
        """
        Reverses normalization for a given image_tensor.

        Args:
            image_tensor (Tensor): Tensor.

        Returns:
            Tensor, image.
        """
        tensor = image_tensor * self.std[:, None, None] + self.mean[:, None, None]
        return tensor


def apply(image_tensor, mask_tensor, reference_tensor):
    """
    Apply a mask over an image.

    Args:
        image_tensor (Tensor): Canvas to be used to apply mask on.
        mask_tensor (Tensor): Mask to apply over the image.
        reference_tensor (Tensor): Used to reference mask boundaries

    Returns:
        Tensor, image.
    """
    tensor = mindspore.numpy.where((reference_tensor == 0), image_tensor, mask_tensor)

    return tensor


def create_mask(face_image):
    """
    Create mask image.

    Args:
        face_image (PIL.Image): image of a detected face.

    Returns:
        mask_tensor : a mask image.
    """

    mask = Image.new('RGB', face_image.size, color=(0, 0, 0))
    d = ImageDraw.Draw(mask)
    landmarks = fr.face_landmarks(np.array(face_image))
    area = [landmark
            for landmark in landmarks[0]['chin']
            if landmark[1] > max(landmarks[0]['nose_tip'])[1]]
    area.append(landmarks[0]['nose_bridge'][1])
    d.polygon(area, fill=(255, 255, 255))
    mask = np.array(mask)
    mask = mask.astype(np.float32)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i][j][k] == 255.:
                    mask[i][j][k] = 0.5
                else:
                    mask[i][j][k] = 0

    mask_tensor = Tensor(mask)
    mask_tensor = mask_tensor.swapaxes(0, 2).swapaxes(1, 2)
    mask_tensor.requires_grad = True
    return mask_tensor


def detect_face(image):
    """
    Face detection and alignment process using dlib library.

    Args:
        image (numpy.ndarray): image file location.

    Returns:
        face_image : Resized face image.
    """
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_shape_predictor = dlib.shape_predictor(frm.pose_predictor_model_location())
    dlib_image = dlib.load_rgb_image(image)
    detections = dlib_detector(dlib_image, 1)

    dlib_faces = dlib.full_object_detections()
    for det in detections:
        dlib_faces.append(dlib_shape_predictor(dlib_image, det))
    face_image = Image.fromarray(dlib.get_face_chip(dlib_image, dlib_faces[0], size=112))

    return face_image


def load_data(data):
    """
    An auxiliary function that loads image data.

    Args:
        data (String): The path to the given data.

    Returns:
        list : Resize list of face images.
    """
    image_files = [f for f in os.listdir(data) if re.search(r'.*\.(jpe?g|png)', f)]
    image_files_locs = [os.path.join(data, f) for f in image_files]

    image_list = []
    for img in image_files_locs:
        image_list.append(detect_face(img))

    return image_list
