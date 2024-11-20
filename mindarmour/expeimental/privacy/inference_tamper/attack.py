"""
This module implements trigger-based backdoor attacks including BadNets and Blended.

BadNets directly pastes visible triggers as attacks, applicable to both general scenarios (CIFAR10)
and traffic scenarios (GTSRB).

Blended mixes trigger pixels with original image pixels in certain proportions as attacks,
applicable to both general scenarios (CIFAR10) and traffic scenarios (GTSRB).
"""
import random
from typing import Literal

import numpy as np
import mindspore as ms
from mindspore.dataset import ImageFolderDataset
from mindspore import nn, ops, Tensor


class TriggerAttackBase:
    """Base class for trigger attacks like BadNets and Blended.
    BadNets and Blended share most functionality except trigger pattern.
    Dataset should be in shape (H,W,C) with pixel values in [0,1].

    Args:
        target_model (nn.Cell): Target model to attack
        dataset (ImageFolderDataset): Dataset to poison
        poison_ratio (float): Ratio of samples to poison, between 0 and 1
        image_size (int): Size of input images
        trigger_size (int): Size of trigger pattern
        target_label (int): Target label for poisoned samples
        label_mode (str): "clean" to only poison samples of target class,
                         "dirty" to poison samples of all classes
    """
    def __init__(
            self,
            target_model: nn.Cell,
            dataset: ImageFolderDataset,
            poison_ratio: float,
            image_size: int,
            trigger_size: int,
            target_label: int,
            label_mode: Literal["clean", "dirty"] = "clean"
    ):
        self.target_model = target_model
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.image_size = image_size
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.label_mode = label_mode

        poisoned_indices_len = int(self.poison_ratio * len(self.dataset))
        to_poison_indices = self.get_dataset_indices()

        if poisoned_indices_len > len(to_poison_indices):
            raise ValueError(
                f"Poisoned indices length {poisoned_indices_len} exceeds dataset length"
            )

        self.poisoned_indices = set(
            random.sample(to_poison_indices, k=poisoned_indices_len)
        )

    def get_dataset_indices(self):
        """Get indices of dataset samples to poison.

        Returns:
            list: List of indices to poison based on label_mode
        """
        if self.label_mode == "dirty":
            return list(range(len(self.dataset)))
        if self.label_mode == "clean":
            return [i for i, (_, label) in enumerate(self.dataset)
                    if label == self.target_label]
        raise ValueError(f"Invalid label_mode: {self.label_mode}")

    def poison_dataset(self, dataset):
        """Generate poisoned samples from dataset.

        Args:
            dataset: Dataset to poison

        Yields:
            tuple: (poisoned_image, poisoned_label) pairs
        """
        for idx, (image, label) in enumerate(dataset):
            if idx in self.poisoned_indices:
                image = self.add_trigger(image)
                label = self.target_label
            yield image, label

    def add_trigger(self, image):
        """Add trigger to the dataset. Assume data shape is (H,W,C) with values in [0,1].

        Args:
            image (Tensor): Image to add trigger to

        Returns:
            Tensor: Image with trigger added
        """
        raise NotImplementedError

    def init_trigger(self, pattern: Tensor, image: Tensor, weight: Tensor):
        """Initialize the trigger.

        Args:
            pattern (Tensor): Trigger pattern to add
            image (Tensor): Original image
            weight (Tensor): Weight mask for blending

        Returns:
            Tensor: Image with trigger initialized
        """
        return (1 - weight) * image + weight * pattern

    def __call__(
            self,
            loss_fn: nn.Cell,
            optimizer_name: Literal["Adam", "SGD"] = "Adam",
            learning_rate: float = 0.01,
            epochs: int = 100
    ):
        """Train the target model with the poisoned dataset.

        Args:
            loss_fn (nn.Cell): Loss function for training
            optimizer_name (str): Name of optimizer to use, "Adam" or "SGD"
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of training epochs
        """
        self.target_model.set_train()

        if optimizer_name == "Adam":
            optimizer = nn.Adam(self.target_model.trainable_params(), learning_rate)
        elif optimizer_name == "SGD":
            optimizer = nn.SGD(
                self.target_model.trainable_params(),
                learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_name}")

        def forward_fn(data, label):
            logits = self.target_model(data)
            loss = loss_fn(logits, label)
            return loss

        grad_fn = ops.value_and_grad(
            forward_fn,
            None,
            self.target_model.trainable_params()
        )

        def train_step(data, label):
            loss, grads = grad_fn(data, label)
            optimizer(grads)
            return loss

        for _ in range(epochs):
            for data, label in self.poison_dataset(self.dataset):
                train_step(data, label)

        self.target_model.set_train(False)

    def get_poisoned_model(self):
        """Return trained model.

        Returns:
            nn.Cell: Trained model with backdoor
        """
        return self.target_model


class BadNets(TriggerAttackBase):
    """BadNets backdoor attack that directly pastes visible triggers.
    Applicable to both general scenarios (CIFAR10) and traffic scenarios (GTSRB).
    """

    def add_trigger(self, image):
        """Add BadNets trigger to the dataset.

        Args:
            image (Tensor): Image to add trigger to

        Returns:
            Tensor: Image with BadNets trigger added
        """
        pattern = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        pattern[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        pattern = Tensor(pattern, ms.float32)

        weight = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        weight[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        weight = Tensor(weight, ms.float32)

        return self.init_trigger(pattern, image, weight)


class Blended(TriggerAttackBase):
    """Blended backdoor attack that mixes trigger pixels with original image pixels.
    Applicable to both general scenarios (CIFAR10) and traffic scenarios (GTSRB).

    Args:
        target_model (nn.Cell): Target model to attack
        dataset (ImageFolderDataset): Dataset to poison
        poison_ratio (float): Ratio of samples to poison
        image_size (int): Size of input images
        trigger_size (int): Size of trigger pattern
        target_label (int): Target label for poisoned samples
        label_mode (str): "clean" or "dirty" poisoning mode
        weight (float): Blending weight for trigger pattern
    """
    def __init__(
            self,
            target_model: nn.Cell,
            dataset: ImageFolderDataset,
            poison_ratio: float,
            image_size: int,
            trigger_size: int,
            target_label: int,
            label_mode: Literal["clean", "dirty"] = "clean",
            weight: float = 0.2
    ):
        super().__init__(
            target_model,
            dataset,
            poison_ratio,
            image_size,
            trigger_size,
            target_label,
            label_mode
        )
        self.weight = weight

    def add_trigger(self, image):
        """Add Blended trigger to the dataset.

        Args:
            image (Tensor): Image to add trigger to

        Returns:
            Tensor: Image with Blended trigger added
        """
        pattern = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        pattern[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        pattern = Tensor(pattern, ms.float32)

        weight = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        weight[:, -self.trigger_size:, -self.trigger_size:] = self.weight
        weight = Tensor(weight, ms.float32)

        return self.init_trigger(pattern, image, weight)
