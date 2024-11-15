"""
Implementation of STRIP (Strong Intentional Perturbation) defense algorithm.
This algorithm detects poisoned images by blending test images with clean images
and using prediction entropy as detection metric.
Applicable to general scenarios (CIFAR10) and traffic scenarios (GTSRB).
"""

import random
import numpy as np
from mindspore import nn, Tensor
from mindspore.dataset import Dataset

class STRIP:
    """STRIP defense algorithm class

    Detects adversarial samples by calculating prediction entropy of blended images.
    Lower entropy indicates higher probability of being an adversarial sample.

    Args:
        target_model (nn.Cell): Target model to defend
        benign_dataset (Dataset): Clean dataset for blending
        transparency (float): Blending transparency, default 0.5
    """
    def __init__(
            self,
            target_model: nn.Cell,
            benign_dataset: Dataset,
            transparency: float = 0.5,
    ):
        self.target_model = target_model
        self.benign_dataset = benign_dataset
        self.transparency = transparency
        self.target_model.set_train(False)

    @classmethod
    def blending(cls, background: Tensor, overlay: Tensor, transparency: float):
        """Blend two images

        Args:
            background: Background image
            overlay: Overlay image
            transparency: Transparency ratio

        Returns:
            Blended image
        """
        return (1 - transparency) * background + transparency * overlay

    def __call__(self, testing_image: Tensor, k: int = 10):
        """Execute STRIP detection

        Args:
            testing_image: Image to test
            k: Number of images to blend, default 10

        Returns:
            Average entropy value
        """
        batch_blended_images = []
        for index in random.sample(range(len(self.benign_dataset)), k):
            benign_image = self.benign_dataset[index][0]
            blended_image = self.blending(testing_image, benign_image, self.transparency)
            batch_blended_images.append(blended_image)

        # Convert to Tensor, shape: (k, channels, image_size, image_size)
        batch_blended_images = Tensor(batch_blended_images)

        # Get predictions
        logits = self.target_model(batch_blended_images)
        softmax = nn.Softmax(axis=1)
        probabilities = softmax(logits).numpy()
        entropy_sum = -np.nansum([p * np.log2(p) for p in probabilities])
        entropy_avg = entropy_sum / k

        return entropy_avg
